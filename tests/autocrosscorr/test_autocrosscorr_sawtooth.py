import numpy as np
from scipy import signal
from wela.autocrosscorr import autocrosscorr
from wela.plotting import plot_replicate_array

nr = 200
nt = 500
period = 8
t = np.linspace(0, period * 4, nt)
ts = np.tile(t, nr).reshape((nr, nt))
y = signal.sawtooth(t * 2 * np.pi / period + 2 * np.pi * np.random.rand(nr, 1))
data = 10 + y + np.random.normal(0, 0.3, y.shape)

# autocorrelation
ac, lags = autocrosscorr(data, stationary=False)
plot_replicate_array(
    ac,
    t=lags * np.median(np.diff(t)) / period,
    xlabel="lag in periods",
    ylabel="correlation",
)
