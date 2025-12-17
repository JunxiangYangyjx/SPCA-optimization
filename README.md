# SPCA-optimization
FEM simulation based optimization algorithm to optimize the SPCA.

maxwell_automation_toolkit.py serves as the backend execution engine, encapsulating low-level pyaedt interactions to handle specific "manual" tasks like 3D modeling, simulation setup, and result extraction within Ansys Maxwell. In contrast, optimization_main_final(1).py acts as the high-level "brain", utilizing the pymoo library to run the NSGA-II algorithm, where it generates design parameters and orchestrates the toolkit to evaluate them, ultimately identifying the optimal coil configurations.

To deploy this framework, a Windows operating system with a valid installation of Ansys Electronics Desktop (e.g., version 2023.1) is required, as the automation relies on the Windows COM interface to control the Maxwell solver. The Python environment requires numpy and pandas for data processing, pymoo for the optimization logic, and pyaedt for the direct simulation interface. The codebase is designed for robustness in long-running tasks, featuring built-in error handling, automatic session management, and rigorous resource release mechanisms to prevent memory leaks during batch simulations.
