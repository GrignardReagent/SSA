#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils.data_processing import _ensure_numpy, _safe_slice

############# Autocorrelation and Cross-correlation ##############
def autocrosscorr(
    yA,
    yB=None,
    stationary=False,
    normalised=True,
    only_pos=False,
):
    """
    Calculate normalised auto- or cross-correlations as a function of lag.

    Lag is given in multiples of the unknown time interval between data
    points, and normalisation is by the product of the standard
    deviation over time for each replicate for each variable.

    For the cross-correlation between sA and sB, the closest peak to zero
    lag should be in the positive lags if sA is delayed compared to
    signal B and in the negative lags if sA is advanced compared to
    signal B.

    Parameters
    ----------
    yA: array
        An array of signal values, with each row a replicate measurement
        and each column a time point.
    yB: array (required for cross-correlation only)
        An array of signal values, with each row a replicate measurement
        and each column a time point.
    stationary: boolean
        If True, the underlying dynamic process is assumed to be
        stationary with the mean a constant, estimated from all
        data points.
    normalised: boolean (optional)
        If True, normalise the result for each replicate by the standard
        deviation over time for that replicate.
    only_pos: boolean (optional)
        If True, return results only for positive lags.

    Returns
    -------
    corr: array
        An array of the correlations with each row the result for the
        corresponding replicate and each column a time point
    lags: array
        A 1D array of the lags in multiples of the unknown time interval

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    Define a sine signal with 200 time points and 333 replicates

    >>> t = np.linspace(0, 4, 200)
    >>> ts = np.tile(t, 333).reshape((333, 200))
    >>> s = 3*np.sin(2*np.pi*ts + 2*np.pi*np.random.rand(333, 1))

    Find and plot the autocorrelaton

    >>> ac, lags = autocrosscorr(s)
    >>> plt.figure()
    >>> plt.plot(lags, np.nanmean(ac, axis=0))
    >>> plt.show()

    Reference
    ---------
    Dunlop MJ, Cox RS, Levine JH, Murray RM, Elowitz MB (2008). Regulatory
    activity revealed by dynamic correlations in gene expression noise.
    Nat Genet, 40, 1493-1498.
    """
    # number of replicates & number of time points
    nr, nt = yA.shape
    # autocorrelation
    if yB is None:
        yB = yA
    # find deviation from the mean
    dyA, stdA = _dev(yA, nr, nt, stationary)
    dyB, stdB = _dev(yB, nr, nt, stationary)
    # calculate correlation
    # lag r runs over positive lags
    pos_corr = np.nan * np.ones(yA.shape)
    for r in range(nt):
        # Vectorized calculation over time
        segment_A = dyA[:, :nt-r]
        segment_B = dyB[:, r:]
        pos_corr[:, r] = np.nanmean(segment_A * segment_B, axis=1)
        
    # lag r runs over negative lags
    # use corr_AB(-k) = corr_BA(k)
    neg_corr = np.nan * np.ones(yA.shape)
    for r in range(nt):
        segment_B = dyB[:, :nt-r]
        segment_A = dyA[:, r:]
        neg_corr[:, r] = np.nanmean(segment_B * segment_A, axis=1)
        
    if normalised:
        # normalise by standard deviation
        pos_corr = pos_corr / stdA / stdB
        neg_corr = neg_corr / stdA / stdB
    # combine lags
    lags = np.arange(-nt + 1, nt)
    corr = np.hstack((np.flip(neg_corr[:, 1:], axis=1), pos_corr))
    # return correlation and lags
    if only_pos:
        return corr[:, int(lags.size / 2) :], lags[int(lags.size / 2) :]
    else:
        return corr, lags

def _dev(y, nr, nt, stationary=False):
    # calculate deviation from the mean
    if stationary:
        # mean calculated over time and over replicates
        dy = y - np.nanmean(y)
    else:
        # mean calculated over replicates at each time point
        dy = y - np.nanmean(y, axis=0).reshape((1, nt))
    # standard deviation calculated for each replicate
    stdy = np.sqrt(np.nanmean(dy**2, axis=1).reshape((nr, 1)))
    return dy, stdy

def calculate_autocorrelation(df, stationary=False):
    """
    Calculate autocorrelation for stress and normal conditions in the dataset.
    Parameters:
    - df: DataFrame with 'label' column and time series data, or numpy array
    - stationary: boolean. If True, assume stationarity (time-average). 
                  If input has only 1 replicate, forced to True.
    Returns:
    - Dictionary with autocorrelation results for available conditions
    """
    results = {}
    stress_data = None
    normal_data = None

    # Check if input is DataFrame with 'label' column
    is_labeled_df = isinstance(df, pd.DataFrame) and 'label' in df.columns

    if is_labeled_df:
        # Separate by label (0 = stress, 1 = normal)
        stress_df = df[df['label'] == 0]
        if not stress_df.empty:
            stress_data = _ensure_numpy(stress_df.drop('label', axis=1))
        
        normal_df = df[df['label'] == 1]
        if not normal_df.empty:
            normal_data = _ensure_numpy(normal_df.drop('label', axis=1))
    else:
        # If not labeled DataFrame, treat all as stress data
        stress_data = _ensure_numpy(df)

    # Process stress data
    if stress_data is not None:
        # Ensure 2D (num_replicates, time_points)
        if stress_data.ndim == 1:
            stress_data = stress_data.reshape(1, -1)
            
        # Check if single replicate, force stationary assumption
        use_stationary = stationary or (stress_data.shape[0] == 1)

        stress_ac, stress_lags = autocrosscorr(stress_data, stationary=use_stationary)
        results.update({
            'stress_ac': stress_ac,
            'stress_lags': stress_lags
        })
    
    # Process normal condition (label = 1) - only if it exists
    if normal_data is not None:
        # Ensure 2D
        if normal_data.ndim == 1:
            normal_data = normal_data.reshape(1, -1)
            
        # Check if single replicate
        use_stationary = stationary or (normal_data.shape[0] == 1)

        normal_ac, normal_lags = autocrosscorr(normal_data, stationary=use_stationary)
        results.update({
            'normal_ac': normal_ac,
            'normal_lags': normal_lags
        })
        
    return results

def calculate_ac_time_interp1d(ac_values, lags):
    """
    Interpolate to find the *first* autocorrelation time where autocorrelation drops below 1/np.exp(1).
    Robust against noise in the tail of the AC function.
    
    Parameters:
    - ac_values: Autocorrelation values.
    - lags: Corresponding lags.
    
    Returns:
    - ac_time: Interpolated autocorrelation time.
    """
    try:
        # Only use positive lags and corresponding autocorrelation values
        positive_mask = lags >= 0
        pos_lags = lags[positive_mask]
        pos_ac = ac_values[positive_mask]
        
        target = 1/np.e
        
        # Check if we have enough data
        if len(pos_ac) < 2:
            return np.nan

        # Check if it starts below target (unlikely but possible)
        if pos_ac[0] < target:
            return pos_lags[0]
            
        # Find indices where AC drops below target
        # Use np.where to find all indices where condition is met
        below_target_indices = np.where(pos_ac < target)[0]
        
        if len(below_target_indices) == 0:
            # print(f"Warning: AC never drops below 1/e ({target:.3f}). Min AC: {np.min(pos_ac):.3f}")
            return np.nan
            
        # Get the first index where it drops below
        idx = below_target_indices[0]
        
        # Ensure idx is valid for interpolation (needs idx-1)
        if idx == 0:
             return pos_lags[0]

        # Linear interpolation between idx-1 and idx
        # y = mx + c
        # x = (y - c) / m
        
        y1 = pos_ac[idx-1]
        y2 = pos_ac[idx]
        x1 = pos_lags[idx-1]
        x2 = pos_lags[idx]
        
        if y1 == y2:
            return x1 # Should not happen unless flat line exactly at transition
            
        # Interpolate x for y=target
        ac_time = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        
        return ac_time
            
    except Exception as e:
        print(f"Error calculating autocorrelation time: {e}")
        return np.nan

def calculate_ac_from_params(rho, d, sigma_b, sigma_u, t_ac, ac=np.exp(-1)):
    """
    Calculate the autocorrelation at a specific time t using the parameters rho, d, sigma_b, and sigma_u.
    
    Parameters:
    - rho: Autocorrelation coefficient.
    - d: Parameter related to the system dynamics.
    - sigma_b: Baseline noise level.
    - sigma_u: Uncorrelated noise level.
    - t_ac: Time at which to calculate the autocorrelation.
    - ac: Target autocorrelation value, default is exp(-1) for steady state.
    
    Returns:
    - Autocorrelation value at time t.
    
    """
    sigma_sum = sigma_b + sigma_u
    numerator = d * np.exp(d - (sigma_sum) * t_ac) * rho * sigma_u - sigma_sum * ((-d**2) + rho * sigma_u + (sigma_sum ** 2))
    denominator = (d - sigma_sum) * (rho * sigma_u + d * sigma_sum + (sigma_sum**2))
    return np.exp(-d * t_ac) * numerator / denominator