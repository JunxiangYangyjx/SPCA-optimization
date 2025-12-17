

import os
import time
import traceback
import numpy as np
import pandas as pd
import logging

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from maxwell_automation_toolkit import MaxwellAutomation

# --- 1. User Configuration Section (from uploaded file) ---
PROJECT_NAME = "Coil_OPT"
DESIGN_NAME_PREFIX = "Opt_Design"
ANSYS_VERSION = "2023.1"
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\OPT"

RMS_TX_CURRENT = 2.0; RMS_TX_EXPR = f"{RMS_TX_CURRENT}*sqrt(2) A"; ANG_TX_EXPR = "0deg"
SIMULATION_FREQUENCY_HZ = 200e3; SIM_FREQ_STR = f"{SIMULATION_FREQUENCY_HZ/1e3}kHz"
Z_GAP_TX_RX = 120.0; COPPER_MATERIAL = "copper"; R1_OUTER_RADIUS = 140.0; H_THICKNESS = 0.035
MIN_INNER_RADIUS = 40.0; NUM_TC_COILS = 1; Z_SEPARATION_TC = -1
POPULATION_SIZE = 20; N_GENERATIONS = 30; MIN_GEOMETRY_DIMENSION = 0.5

# --- 2. LHS Initial Population Generation ---
class LHSValidGeometrySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0.0); lhs_sampler = LHS(); n_candidates = n_samples * 50; candidates_pop = lhs_sampler(problem, n_candidates, **kwargs)
        valid_count = 0
        for individual in candidates_pop:
            if valid_count >= n_samples: break
            candidate_vars = individual.X; N = int(np.round(candidate_vars[0])); w_outer = candidate_vars[1]; k_w = candidate_vars[2]; s_outer = candidate_vars[3]; k_s = candidate_vars[4]
            w_list = np.array([w_outer * (k_w ** i) for i in range(N)]); s_list = np.array([s_outer * (k_s ** i) for i in range(N - 1)]) if N > 1 else np.array([])
            if np.any(w_list < MIN_GEOMETRY_DIMENSION) or np.any(s_list < MIN_GEOMETRY_DIMENSION): continue
            r_inner = R1_OUTER_RADIUS - np.sum(w_list) - np.sum(s_list)
            if r_inner > MIN_INNER_RADIUS: X[valid_count] = candidate_vars; valid_count += 1
        if valid_count < n_samples: print(f"Warning: Only find {valid_count}/{n_samples} effective initial solutions.")
        print(f"--- {valid_count} LHS-based geometry-valid initial designs successfully generated ---"); return X[:valid_count]

# --- 3. Objective function ---
def evaluate_design(variables, gen_idx, ind_idx, automation):
    N = int(np.round(variables[0])); w_outer = np.round(variables[1], 3); k_w = np.round(variables[2], 3); s_outer = np.round(variables[3], 3); k_s = np.round(variables[4], 3)
    design_name = f"{DESIGN_NAME_PREFIX}_G{gen_idx}_I{ind_idx}"; print(f"\n{'='*25} [Gen {gen_idx} | Ind {ind_idx}] | Design: {design_name} {'='*25}"); print(f"--- Core Variables (Rounded): N={N}, W_outer={w_outer:.3f}, k_w={k_w:.3f}, S_outer={s_outer:.3f}, k_s={k_s:.3f}")
    w_list = [w_outer * (k_w ** i) for i in range(N)]; s_list = [s_outer * (k_s ** i) for i in range(N - 1)] if N > 1 else []
    objectives_to_return = [float('inf'), float('inf')]
    if any(w < MIN_GEOMETRY_DIMENSION for w in w_list) or any(s < MIN_GEOMETRY_DIMENSION for s in s_list):
        print("--- Invalid geometry detected. Skipping simulation. ---")
    else:
        try:
            automation.app.insert_design(design_name, solution_type="EddyCurrent"); automation.set_active_design(design_name)
            tc_coil_names = [f"Tc{i+1}" for i in range(NUM_TC_COILS)]; all_coil_names = ["Tx", "Rx"] + tc_coil_names; coil_z_positions = {"Tx": 0, "Rx": Z_GAP_TX_RX}
            for i, name in enumerate(tc_coil_names): coil_z_positions[name] = Z_SEPARATION_TC * (i + 1)
            for name in all_coil_names:
                rms, ang = (RMS_TX_EXPR, ANG_TX_EXPR) if name != "Rx" else ("0A", "0deg")
                automation.create_complete_coil(N, R1_OUTER_RADIUS, s_list, w_list, H_THICKNESS, name, coil_z_positions[name], COPPER_MATERIAL, rms, ang)
            automation.create_region(); automation.setup_analysis(SIM_FREQ_STR, "Setup1", "Matrix1", all_coil_names)
            if not automation.analyze(): raise RuntimeError("Simulation analysis fails!")
            matrix_results = automation.extract_matrix_results(all_coil_names); Mtr = matrix_results.get("L(Tx,Rx)", 0)
            Ploss_total = automation.extract_total_ohmic_loss(all_coil_names, SIM_FREQ_STR, RMS_TX_EXPR, ANG_TX_EXPR)
            print(f"--- Results: Mtr = {Mtr:.8f} H, Ploss_total (Ohmic) = {Ploss_total:.4f} W"); objectives_to_return = [-Mtr, Ploss_total]
        except Exception as e:
            print(f"!!!!!! ERROR during evaluation of Design {design_name} !!!!!! "); traceback.print_exc(); objectives_to_return = [float('inf'), float('inf')]
        finally: automation.cleanup_design(design_name)
    record = {'Generation': gen_idx, 'Individual': ind_idx, 'Mtr_H': -objectives_to_return[0] if np.isfinite(objectives_to_return[0]) else 0, 'Ploss_total_W': objectives_to_return[1] if np.isfinite(objectives_to_return[1]) else float('inf'), 'N_turns': N, 'W_outer_mm': w_outer, 'k_w_ratio': k_w, 'S_outer_mm': s_outer, 'k_s_ratio': k_s}
    return objectives_to_return, record

# --- 4. Pymoo problem definition ---
class AnsysProblem(Problem):
    def __init__(self, excel_path):
        n_vars = 5
        xl = np.array([9, 2.0, 0.85, 0.8, 1.0])
        xu = np.array([16, 5.0, 1.0,  2.5, 1.15])
        super().__init__(n_var=n_vars, n_obj=2, n_constr=0, xl=xl, xu=xu)
        self.excel_path = excel_path
        self.eval_count = 0
        self.aedt_app = None

    def _start_aedt(self):
        if self.aedt_app is None:
            project_name_with_gen = f"{PROJECT_NAME}_G{self.eval_count // POPULATION_SIZE}"
            self.aedt_app = MaxwellAutomation(projectname=project_name_with_gen, ansys_version=ANSYS_VERSION, non_graphical=True, new_session=True)
            if self.aedt_app and hasattr(self.aedt_app.app, "logger"):
                self.aedt_app.app.logger._global.setLevel(logging.WARNING)
                print("--- PyAEDT console log level set to WARNING. INFO messages will be hidden. ---")

    def _stop_aedt(self):
        if self.aedt_app:
            self.aedt_app.release()
            self.aedt_app = None
            print("\n--- AEDT Session Released for Periodic Restart ---\n")

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        results_this_generation = []
        self._start_aedt()
        for i in range(x.shape[0]):
            gen_idx = self.eval_count // POPULATION_SIZE
            ind_idx = self.eval_count % POPULATION_SIZE
            objectives, record = evaluate_design(x[i], gen_idx, ind_idx, self.aedt_app)
            self.eval_count += 1
            F.append(objectives)
            results_this_generation.append(record)
            print("--- Simulation finished. Pausing for 5 seconds... ---")
            time.sleep(5)
            
        df_batch = pd.DataFrame(results_this_generation)
        if not os.path.exists(self.excel_path):
            df_batch.to_excel(self.excel_path, index=False, engine='openpyxl')
        else:
            with pd.ExcelWriter(self.excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df_batch.to_excel(writer, header=False, index=False, startrow=writer.sheets['Sheet1'].max_row)
        print(f"\n--- Batch Write: Appended {len(results_this_generation)} results to log file. ---\n")
        
        is_not_last_total = (self.eval_count < POPULATION_SIZE * N_GENERATIONS)
        if is_not_last_total:
            current_gen = self.eval_count // POPULATION_SIZE - 1
            print("\n" + "*"*30); print(f"--- Generation {current_gen} Finished. Restarting AEDT... ---"); print("*"*30 + "\n")
            self._stop_aedt()
            print("--- Resting for 2 minutes before next generation... ---")
            time.sleep(120)

        out["F"] = np.array(F)

# --- 5. Main ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    excel_file = os.path.join(OUTPUT_DIR, "full_optimization_log.xlsx")
    if os.path.exists(excel_file): os.remove(excel_file)

    problem = AnsysProblem(excel_path=excel_file)
    
    try:
        algorithm = NSGA2(pop_size=POPULATION_SIZE, sampling=LHSValidGeometrySampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
        termination = ("n_gen", N_GENERATIONS)
        print("\n" + "*"*80 + "\n--- Multi-Objective Optimization Started (V34 - Final Robust Version) ---\n" + "*"*80 + "\n")
        start_time = time.time()
        res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)
        end_time = time.time(); print(f"\n--- Optimization Finished! Total Time: {(end_time - start_time)/3600:.2f} hours ---")
    except Exception as e:
        print("An unexpected error occurred during the optimization."); traceback.print_exc()
    finally:
        if problem and problem.aedt_app:
            print("Final cleanup: Releasing Ansys AEDT session...")
            problem.aedt_app.release()
