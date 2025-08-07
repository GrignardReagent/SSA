#!/usr/bin/python
import numpy as np
from scipy.interpolate import interp1d

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
        prods = [dyA[:, t] * dyB[:, t + r] for t in range(nt - r)]
        pos_corr[:, r] = np.nanmean(prods, axis=0)
    # lag r runs over negative lags
    # use corr_AB(-k) = corr_BA(k)
    neg_corr = np.nan * np.ones(yA.shape)
    for r in range(nt):
        prods = [dyB[:, t] * dyA[:, t + r] for t in range(nt - r)]
        neg_corr[:, r] = np.nanmean(prods, axis=0)
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

# Function to calculate autocorrelation for a dataset
def calculate_autocorrelation(df):
    # Separate by label (0 = stress, 1 = normal)
    stress_df = df[df['label'] == 0]
    normal_df = df[df['label'] == 1]
    
    # Remove 'label' column and convert to numpy array
    stress_data = stress_df.drop('label', axis=1).values
    normal_data = normal_df.drop('label', axis=1).values
    
    # Calculate autocorrelation
    stress_ac, stress_lags = autocrosscorr(stress_data)
    normal_ac, normal_lags = autocrosscorr(normal_data)
    
    return {
        'stress_ac': stress_ac,
        'stress_lags': stress_lags,
        'normal_ac': normal_ac,
        'normal_lags': normal_lags
    }

def calculate_ac_time_interp1d(ac_values, lags):
    """
    Interpolate to find the autocorrelation time where autocorrelation = 1/np.exp(1).
    
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
        
        # Check if 1/e is within the range of autocorrelation values
        threshold = 1/np.e
        if threshold < np.min(pos_ac) or threshold > np.max(pos_ac):
            print(f"Warning: 1/e threshold ({threshold:.3f}) is outside the range of AC values [{np.min(pos_ac):.3f}, {np.max(pos_ac):.3f}]")
            return np.nan
        
        # Interpolate using interp1d with correct parameter order (x, y)
        f_interp = interp1d(pos_ac, pos_lags, kind='linear', bounds_error=False, fill_value=np.nan)
        ac_time = f_interp(threshold)
        return ac_time
            
    except Exception as e:
        print(f"Error calculating autocorrelation time: {e}")
        return np.nan