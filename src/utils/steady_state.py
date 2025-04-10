import pandas as pd
import os

#!/user/bin/python

def find_steady_state(parameter_set: dict):
    """
    Determine the time point when the system reaches steady state, this is usually 10/degradation rate, so this function take the parameter_set as the input and return the steady state time.
    
    Parameters:
        parameter_set (dict): Parameter set for the simulation.
        
    Returns:
        steady_state_time (float): Time when steady state is reached.
    """
    # Extract the degradation rate from the parameter set
    degradation_rate = parameter_set['d']
    steady_state_time = 10 / degradation_rate  # Time to reach steady state is usually 10 / degradation rate

    return steady_state_time, int(steady_state_time) # time, index

# save only the steady state part of the simulated time series
def save_steady_state(filename: str, parameter_sets: list, time_points, save_path: str = None):
    """
    Save only the steady state part of the simulated time series to a new file.
    
    Parameters:
        filename (str): Path to the file containing the time series data.
        parameter_sets (list): List of parameter sets used for the simulation, for both conditions.
        time_points (list): List of time points for the simulation.
        save_path (str): Path to save the steady state time series. If not None and not a file name, saves the file as "{filename}_SS.csv".
        
    Returns:
        steady_state_series (pd.DataFrame): Steady state part of the time series.
    """
    # Read the file as a pandas DataFrame
    df_results = pd.read_csv(filename)

    min_d = min([param_set['d'] for param_set in parameter_sets]) 
    # get the time point after which the system reaches steady state 
    steady_state_time = int(10 / min_d)
    remaining_time_points = time_points[(steady_state_time-1):]
    # Extract the steady state part of the time series, but remember we need to retain the label column when slicing
    steady_state_series = df_results.loc[:, ['label'] + list(df_results.columns[steady_state_time:])]

    # TODO: Need to be able to handle situations where we have multiple conditions e.g. normal and stress
    # Save the steady state series to a new file (optional)
    if save_path is not None:
        # check if save_path is a file name or directory
        if save_path.endswith('.csv'):
            steady_state_filename = save_path
            # create the parent folder for save_path if not existing
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            os.makedirs(save_path, exist_ok=True)
            steady_state_filename = os.path.join(save_path, f"{os.path.splitext(os.path.basename(filename))[0]}_SS.csv")
        print(f"Steady state series saved to {steady_state_filename}")
        
        steady_state_series.to_csv(steady_state_filename, index=False)
    
    return remaining_time_points, steady_state_series
