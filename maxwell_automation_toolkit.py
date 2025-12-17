# filename: maxwell_automation_toolkit.py
# Final Version with Robust Release Method
import os
import time
import csv
import re
import shutil
import traceback
import math
import logging
from pyaedt import Maxwell3d

class MaxwellAutomation:
    def __init__(self, projectname, ansys_version="2023.1", non_graphical=False, new_session=True):
        print("--- Initializing Ansys Maxwell Session... ---")
        try:
            self.app = Maxwell3d(
                projectname=projectname,
                specified_version=ansys_version,
                non_graphical=non_graphical,
                new_desktop_session=new_session
            )
        except Exception as e:
            print(f"--- Initializing AEDT failed with error: {e}. Retrying once... ---")
            time.sleep(5) 
            self.app = Maxwell3d(
                projectname=projectname,
                specified_version=ansys_version,
                non_graphical=non_graphical,
                new_desktop_session=new_session
            )
        
        try:
            file_handler_to_remove = None
            for handler in self.app.logger._global.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    file_handler_to_remove = handler
                    break
            if file_handler_to_remove:
                self.app.logger._global.removeHandler(file_handler_to_remove)
                print("--- PyAEDT file logging handler removed successfully. ---")
        except Exception as e:
            print(f"--- WARNING: Could not remove PyAEDT file logger. Error: {e} ---")

        self.temp_dir = os.path.join(self.app.project_path, "temp_results")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.oEditor = None
        self.oDesign = None
        self.oModuleBoundary = None
        print("--- Ansys Maxwell Session Initialized. ---")

    def set_active_design(self, design_name):
        print(f"--- Setting active design to: {design_name} ---")
        self.app.set_active_design(design_name)
        self.oDesign = self.app.odesign
        self.oEditor = self.app.modeler.oeditor
        self.oModuleBoundary = self.oDesign.GetModule("BoundarySetup")

    def cleanup_design(self, design_name):
        print(f"--- Cleaning up and deleting design: {design_name} ---")
        try:
            if self.app and self.app.odesktop and design_name in self.app.design_list:
                self.app.delete_design(design_name)
        except Exception as e:
            print(f"--- WARNING: Could not delete design {design_name}. It might already be closed. Error: {e} ---")
            
    def release(self):
        if self.app:
            print("--- Releasing AEDT Desktop... ---")
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            try:
                # 优先尝试使用更彻底的关闭方式
                self.app.release_desktop(close_projects=True, close_on_exit=True)
            except TypeError:
                # 如果不支持 close_on_exit 参数，则使用兼容模式
                print("--- INFO: 'close_on_exit' not supported. Using fallback release method. ---")
                self.app.release_desktop(close_projects=True)
            except Exception as e:
                print(f"--- WARNING: An error occurred during AEDT release: {e} ---")
            self.app = None
            print("--- PyAEDT session released. ---")

    # ... 其他所有函数 (create_complete_coil, setup_analysis 等) 保持不变 ...
    def create_complete_coil(self, n, r1, s_list, w_list, h, coil_name, z_offset, material, rms_current, phase):
        print(f"\n--- Creating complete coil: {coil_name} at Z={z_offset}mm ---"); coil_parts, ring_radii, current_radius = [], [], r1
        for i in range(n):
            w = w_list[i]
            outer = self.app.modeler.create_circle("XY", [0,0,z_offset], current_radius, name=f"{coil_name}_O_{i}"); self.oEditor.ChangeProperty(["NAME:AllTabs", ["NAME:Geometry3DCmdTab", ["NAME:PropServers", f"{outer.name}:CreateCircle:1"], ["NAME:ChangedProps", ["NAME:Number of Segments", "Value:=", "58"]]]])
            inner = self.app.modeler.create_circle("XY", [0,0,z_offset], current_radius-w, name=f"{coil_name}_I_{i}"); self.oEditor.ChangeProperty(["NAME:AllTabs", ["NAME:Geometry3DCmdTab", ["NAME:PropServers", f"{inner.name}:CreateCircle:1"], ["NAME:ChangedProps", ["NAME:Number of Segments", "Value:=", "58"]]]])
            self.app.modeler.subtract(outer, inner, keep_originals=False); coil_parts.append(outer.name); ring_radii.append({"outer":current_radius, "inner":current_radius-w})
            if i < n - 1: s = s_list[i]; current_radius -= (w + s)
                        # --- 核心修改: 增加一个0.1秒的微观延时来提高建模稳定性 ---
            time.sleep(0.3)
        gap_width = 8.0; gap_start_y, gap_end_y = ring_radii[-1]["inner"]-1, ring_radii[0]["outer"]+1
        cutter = self.app.modeler.create_rectangle("XY", [-gap_width/2, gap_start_y, z_offset], [gap_width, gap_end_y - gap_start_y], name=f"{coil_name}_cutter"); self.app.modeler.subtract(coil_parts, cutter, keep_originals=False)
        for i in range(n - 1):
            r_curr_in, r_curr_out = ring_radii[i]["inner"], ring_radii[i]["outer"]; r_next_in, r_next_out = ring_radii[i+1]["inner"], ring_radii[i+1]["outer"]; half_gap_sq = (gap_width / 2)**2
            y1_val = r_curr_out**2 - half_gap_sq; y1 = math.sqrt(y1_val) if y1_val > 0 else 0; y2_val = r_curr_in**2 - half_gap_sq; y2 = math.sqrt(y2_val) if y2_val > 0 else 0
            y3_val = r_next_in**2 - half_gap_sq; y3 = math.sqrt(y3_val) if y3_val > 0 else 0; y4_val = r_next_out**2 - half_gap_sq; y4 = math.sqrt(y4_val) if y4_val > 0 else 0
            p1 = [-gap_width/2, y1, z_offset]; p2 = [-gap_width/2, y2, z_offset]; p3 = [gap_width/2, y3, z_offset]; p4 = [gap_width/2, y4, z_offset]
            bridge = self.app.modeler.create_polyline([p1,p2,p3,p4,p1], name=f"{coil_name}_b_{i}", cover_surface=True); coil_parts.append(bridge.name)
        self.app.modeler.unite(coil_parts); sheet_object = self.app.modeler.get_object_from_name(coil_parts[0]); sheet_object.name = f"sheet_{coil_name}"
        self.app.modeler.sweep_along_vector(f"sheet_{coil_name}", [0, 0, h]); final_coil_name = f"coil_{coil_name}"; self.app.modeler.get_object_from_name(f"sheet_{coil_name}").name = final_coil_name; self.app.modeler.get_object_from_name(final_coil_name).material_name = material
        try:
            half_gap_sq = (gap_width / 2)**2
            y_inner_terminal_val = ring_radii[n-1]["inner"]**2 - half_gap_sq; y_inner_terminal = math.sqrt(y_inner_terminal_val) if y_inner_terminal_val > 0 else 0
            y_outer_terminal_val = ring_radii[0]["inner"]**2 - half_gap_sq; y_outer_terminal = math.sqrt(y_outer_terminal_val) if y_outer_terminal_val > 0 else 0
            plane_1 = self.app.modeler.create_rectangle("XY",[-gap_width/2, y_inner_terminal, z_offset], [0.1, w_list[-1]], name=f"plane_1_{coil_name}"); plane_2 = self.app.modeler.create_rectangle("XY",[gap_width/2, y_outer_terminal, z_offset], [-0.1, w_list[0]], name=f"plane_2_{coil_name}")
            self.app.modeler.sweep_along_vector(plane_1.name, [0, 0, 0.3]); self.app.modeler.sweep_along_vector(plane_2.name, [0, 0, 0.3])
            plane_3 = self.app.modeler.create_rectangle("XY",[-gap_width/4, y_inner_terminal, z_offset+0.3],[-gap_width/4,ring_radii[0]["outer"]-ring_radii[n-1]["inner"]],name=f"plane_3_{coil_name}"); plane_4 = self.app.modeler.create_rectangle("XY",[gap_width/2, y_outer_terminal, z_offset+0.3], [-gap_width, w_list[0]], name=f"plane_4_{coil_name}")
            self.app.modeler.sweep_along_vector(plane_3.name, [0, 0, -0.01]); self.app.modeler.sweep_along_vector(plane_4.name, [0, 0, -0.01])
            self.app.modeler.unite([final_coil_name, plane_1.name, plane_2.name, plane_3.name, plane_4.name])
        except Exception as term_e: print(f"--- WARNING: Terminal creation logic failed for {coil_name}. Error: {term_e}")
        print(f"Setting current excitation for {coil_name}..."); self.app.variable_manager.set_variable(f"RMS_{coil_name}", rms_current); self.app.variable_manager.set_variable(f"Ang_{coil_name}", phase); section_name = f"{final_coil_name}_Section1"
        try:
            self.oEditor.Section(["NAME:Selections", "Selections:=", final_coil_name], ["NAME:SectionToParameters", "SectionPlane:=", "YZ"]); time.sleep(0.5); self.app.modeler.separate_bodies(section_name); time.sleep(0.5); target_face_name = f"{section_name}_Separate1"
            if target_face_name not in self.app.modeler.object_names: print(f"--- WARNING: Could not find target excitation face '{target_face_name}'.")
            else:
                all_separated_faces = [n for n in self.app.modeler.object_names if n.startswith(f"{section_name}_Separate")]; faces_to_delete = [n for n in all_separated_faces if n != target_face_name]
                if faces_to_delete: self.app.modeler.delete(faces_to_delete)
                self.oModuleBoundary.AssignCurrent(["NAME:"+coil_name, "Objects:=", [target_face_name], "Phase:=", f"Ang_{coil_name}", "Current:=", f"RMS_{coil_name}", "IsSolid:=", True, "Point out of terminal:=", False]); print(f"Coil {coil_name} created and excited successfully.")
        except Exception as exc_e: print(f"!!!!!! ERROR: Excitation creation logic failed for {coil_name}. Error: {exc_e}")
    def set_initial_mesh_settings(self):
        print("Setting initial mesh..."); oModule = self.oDesign.GetModule("MeshSetup"); oModule.InitialMeshSettings(["NAME:MeshSettings", ["NAME:GlobalSurfApproximation", "CurvedSurfaceApproxChoice:=", "UseSlider", "SliderMeshSettings:=", 4], ["NAME:GlobalCurvilinear", "Apply:=", True], ["NAME:GlobalModelRes", "UseAutoLength:=", True], "MeshMethod:=", "Auto", "UseLegacyFaceterForTauVolumeMesh:=", False, "DynamicSurfaceResolution:=", False, "UseFlexMeshingForTAUvolumeMesh:=", False, "UseAlternativeMeshMethodsAsFallBack:=", True, "AllowPhiForLayeredGeometry:=", False])
    def create_region(self):
        print("Creating custom simulation region..."); self.oEditor.CreateRegion(["NAME:RegionParameters", "+XPaddingType:=", "Percentage Offset", "+XPadding:=", "90", "-XPaddingType:=", "Percentage Offset", "-XPadding:=", "90", "+YPaddingType:=", "Percentage Offset", "+YPadding:=", "90", "-YPaddingType:=", "Percentage Offset", "-YPadding:=", "90", "+ZPaddingType:=", "Percentage Offset", "+ZPadding:=", "150", "-ZPaddingType:=", "Percentage Offset", "-ZPadding:=", "150"], ["NAME:Attributes", "Name:=", "Region", "Flags:=", "", "Color:=", "(143 175 143)", "Transparency:=", 1, "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", "\"vacuum\"", "SurfaceMaterialValue:=", "\"\"", "SolveInside:=", True, "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False, "IsLightweight:=", False])
    def setup_analysis(self, frequency, name, matrix_name, coil_names):
        print("Configuring solution setup..."); self.app.eddy_effects_on([f"coil_{name}" for name in coil_names])
        setup = self.app.create_setup(name=name)
        setup.props["Frequency"] = frequency; setup.props["MaximumPasses"] = 14; setup.props["PercentError"] = 2; setup.update(); oModuleMatrix = self.oDesign.GetModule("MaxwellParameterSetup"); matrix_entries = [["NAME:MatrixEntry", "Source:=", name] for name in coil_names]; oModuleMatrix.AssignMatrix(["NAME:Matrix1", ["NAME:MatrixEntry", *matrix_entries]]); self.set_initial_mesh_settings()
    def analyze(self, setup_name="Setup1"):
        print(f"--- Analyzing {setup_name}, this may take several minutes... ---"); return self.app.analyze_setup(setup_name)
    def extract_matrix_results(self, all_coil_names):
        results = {}; oModule_report = self.oDesign.GetModule("ReportSetup"); export_path = os.path.join(self.temp_dir, "matrix.csv"); report_name = "Temp_Matrix_Report"
        y_comp = sorted(list(set([f"Matrix1.R({c},{c})" for c in all_coil_names] + [f"Matrix1.L({c1},{c2})" for i, c1 in enumerate(all_coil_names) for j, c2 in enumerate(all_coil_names) if j >= i]))); variations = ["Freq:=", ["All"]] + [item for name in all_coil_names for item in ([f"Ang_{name}:=", ["Nominal"], f"RMS_{name}:=", ["Nominal"]])]
        oModule_report.CreateReport(report_name, "EddyCurrent", "Data Table", "Setup1 : LastAdaptive", [], variations, ["X Component:=", "Freq", "Y Component:=", y_comp]); o_module_report = self.oDesign.GetModule("ReportSetup"); o_module_report.ExportToFile(report_name, export_path, False)
        self._wait_for_file(export_path); oModule_report.DeleteReports([report_name])
        with open(export_path, 'r', encoding='utf-8') as f:
            r = csv.reader(f); header, values = next(r), next(r); unit_mult = {'uH': 1e-6, 'nH': 1e-9, 'mH': 1e-3, 'H': 1, 'ohm': 1, 'mohm': 1e-3, 'kohm': 1e3}
            for i in range(1, len(header)):
                col_name = header[i]; match_p = re.search(r'([LR])\((.*?)\)', col_name); match_u = re.search(r'\[(.*?)\]', col_name)
                if match_p: key = f"{match_p.group(1)}({match_p.group(2)})"; unit = match_u.group(1).lower() if match_u else 'ohm'; results[key] = float(values[i]) * unit_mult.get(unit, 1)
        os.remove(export_path); return results
    def extract_total_ohmic_loss(self, all_coil_names, freq_str, rms_expr_str, ang_expr_str):
        print("--- Calculating total ohmic loss... ---"); oModule_field = self.oDesign.GetModule("FieldsReporter"); export_path = os.path.join(self.temp_dir, "total_loss.csv")
        try:
            calc_vars = [];
            for name in all_coil_names: calc_vars.extend([f"Ang_{name}:=", ang_expr_str if name != "Rx" else "0deg"]); calc_vars.extend([f"RMS_{name}:=", rms_expr_str if name != "Rx" else "0A"])
            calc_vars.extend(["Freq:=", freq_str, "Phase:=", "0deg"])
            oModule_field.CopyNamedExprToStack("Ohmic_Loss"); oModule_field.EnterVol("AllObjects"); oModule_field.CalcOp("Integrate")
            oModule_field.CalculatorWrite(export_path, ["Solution:=", "Setup1 : LastAdaptive"], calc_vars)
            self._wait_for_file(export_path, timeout=60)
            with open(export_path, 'r', encoding='utf-8') as f:
                r = csv.reader(f); next(r); total_loss_value = float(next(r)[0])
            os.remove(export_path); print(f"  - Total Ohmic Loss = {total_loss_value:.4f} W"); return total_loss_value
        except Exception as e:
            print(f"!!!!!! ERROR: Failed to calculate total loss. Error: {e}"); traceback.print_exc(); return float('inf')
    def _wait_for_file(self, file_path, timeout=30):
        start_time = time.time()
        while not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            time.sleep(0.5)
            if time.time() - start_time > timeout: raise TimeoutError(f"Timeout waiting for file {file_path}.")