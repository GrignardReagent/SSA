# import the simulation functions
import multiprocessing
import pandas as pd 
import numpy as np
import os
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

def simulate_telegraph_model(parameter_sets, time_points, size, num_cores=None):
    """
    Simulates the telegraph model using Julia implementation.
    Parameters:
        parameter_sets (list): List of parameter dictionaries for each system.
        time_points (numpy array): Array of time points for the simulation.
        size (int): Number of simulations per condition.
        num_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.
    Returns:
        pd.DataFrame: DataFrame containing simulation results with columns for 'label' and time points.
    Usage:
        df_results = simulate_telegraph_model(parameter_sets, time_points, size, num_cores)
        # where parameter_sets is a list of dictionaries with keys like 'sigma_u', 'sigma_b', 'rho', 'd', and 'label'.
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
        jl.seval('using Pkg; Pkg.activate("/home/ianyang/stochastic_simulations/julia"); Pkg.instantiate()')
        jl.seval('using DataFrames, NPZ')
        jl.include("/home/ianyang/stochastic_simulations/julia/simulation/TelegraphSSA.jl")
        jl.seval('using .TelegraphSSA')
        simulate_telegraph_model._julia_initialized = True

    # Python → Julia conversion handled automatically
    jl.parameter_sets = parameter_sets
    jl.time_points = time_points
    
    # Run the simulation in Julia
    jl.seval(f'df_julia = simulate_telegraph_model(parameter_sets, time_points, {size})')
    
    # Convert Julia DataFrame directly to Python
    labels = np.array(jl.seval('Int64.(df_julia.label)'))
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