# import the simulation functions
import multiprocessing
import warnings
import pandas as pd
import numpy as np
import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

def simulate_telegraph_model(parameter_sets, time_points, size, num_cores=None):
    """
    Simulates the telegraph model using Julia implementation.

    The telegraph model is a 2-state stochastic model for gene expression with mRNA
    production and degradation. Simulations use the Gillespie SSA algorithm.

    Parameters:
        parameter_sets (list): List of parameter dictionaries, one per condition.
            Each dict must have keys: 'sigma_u', 'sigma_b', 'rho', 'd', and optionally 'label'.
            - sigma_u: gene activation rate
            - sigma_b: gene deactivation rate
            - rho: mRNA production rate
            - d: mRNA degradation rate
            - label: identifier for the condition (default: 0)
        time_points (numpy.ndarray): Array of time points at which to save mRNA counts.
        size (int): Number of independent trajectories per parameter set.
        num_cores (int, optional): Number of CPU cores to use (must be set before first
            import of juliacall). Defaults to all available cores.

    Returns:
        pd.DataFrame: Simulation results with columns:
            - 'label': parameter set identifier
            - 'time_{t}': mRNA count at time t (one column per time point)

    Example:
        import numpy as np
        from simulation.julia_simulate_telegraph_model import simulate_telegraph_model

        # Define parameters for two conditions
        parameter_sets = [
            {'sigma_u': 1.0, 'sigma_b': 1.0, 'rho': 10.0, 'd': 1.0, 'label': 'control'},
            {'sigma_u': 2.0, 'sigma_b': 1.0, 'rho': 15.0, 'd': 1.0, 'label': 'treatment'},
        ]

        # Time points from 0 to 100 (linear steps of 1)
        time_points = np.arange(0, 100, 1.0)

        # Run 1000 trajectories per condition using all available cores
        df_results = simulate_telegraph_model(parameter_sets, time_points, size=10)

        # Results: df_results has 2000 rows (1000 per condition)
        # and 101 columns: 'label' + 100 time_* columns
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    # print(f"Using {num_cores} cores for Julia simulation.")
    # threads need to be set using an environment variable before Julia starts
    os.environ["JULIA_NUM_THREADS"] = str(num_cores)
    
    from juliacall import Main as jl
    # Initialize Julia environment once (only if not already done)
    if not hasattr(simulate_telegraph_model, '_julia_initialized'):
        print("Initializing Julia environment...")
        
        # 1. Get the directory where this script is located
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Resolve the absolute path to the 'julia' folder relative to this script
        #    Go up 2 levels (src/ -> simulation/) to get to the project root
        project_root = os.path.abspath(os.path.join(current_script_dir, "../../"))
        julia_env_path = os.path.join(project_root, "julia")
        
        # 3. Resolve path to the Julia source file
        julia_src_file = os.path.join(julia_env_path, "simulation/TelegraphSSA.jl")

        # Check if paths exist to provide a helpful error if the structure is wrong
        if not os.path.exists(julia_env_path):
            raise FileNotFoundError(f"Could not find Julia environment at: {julia_env_path}")
        if not os.path.exists(julia_src_file):
            raise FileNotFoundError(f"Could not find Julia source file at: {julia_src_file}")

        # 4. Pass the dynamic paths to Julia (using formatted strings)
        #    We replace backslashes with forward slashes for Windows compatibility
        jl_env_cmd = f'using Pkg; Pkg.activate("{julia_env_path.replace(os.sep, "/")}"); Pkg.instantiate()'
        jl_include_cmd = f'include("{julia_src_file.replace(os.sep, "/")}")'

        jl.seval(jl_env_cmd)
        jl.seval('using DataFrames, NPZ')
        jl.seval(jl_include_cmd)
        jl.seval('using .TelegraphSSA')

        # Warn if Julia started with fewer threads than requested.
        # Julia's thread count is fixed at startup; setting JULIA_NUM_THREADS
        # after juliacall has already been imported has no effect.
        actual_threads = int(jl.seval('Threads.nthreads()'))
        if actual_threads < num_cores:
            warnings.warn(
                f"Requested {num_cores} Julia threads but Julia started with "
                f"{actual_threads}. Julia's thread count is fixed at startup. "
                f"Set JULIA_NUM_THREADS={num_cores} in the environment before "
                f"importing juliacall (i.e. before the first call to this function).",
                RuntimeWarning,
                stacklevel=2,
            )

        simulate_telegraph_model._julia_initialized = True

    # Python → Julia conversion handled automatically
    jl.parameter_sets = parameter_sets
    jl.time_points = time_points
    
    # Run the simulation in Julia
    jl.seval(f'df_julia = simulate_telegraph_model(parameter_sets, time_points, {size})')

    # Convert Julia DataFrame directly to Python
    # Labels can be strings, integers, or any type — don't force conversion
    labels = np.array(jl.seval('df_julia.label'))
    counts_matrix = np.array(jl.seval('Int64.(Matrix(df_julia[:, Not(:label)]))'))

    # Create pandas DataFrame with same column format as Python version
    df_labels = pd.DataFrame(labels, columns=['label'])
    df_counts = pd.DataFrame(counts_matrix, columns=[f"time_{ti}" for ti in time_points])
    df = pd.concat([df_labels, df_counts], axis=1)

    # Check for NaN values before returning
    if df.isna().sum().sum() > 0:
        print("⚠️ Warning: NaN values detected in df!")
        print(df.isna().sum()[df.isna().sum() > 0])

    return df