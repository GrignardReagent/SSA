import numpy as np
from wela.autocrosscorr import autocrosscorr
from wela.plotting import plot_replicate_array

nr = 200
nt = 500
addrandomphase = False
period = 8
t = np.linspace(0, period * 4, nt)
ts = np.tile(t, nr).reshape((nr, nt))
if addrandomphase:
    y = 3 * np.sin(2 * np.pi * ts / period + 2 * np.pi * np.random.rand(nr, 1))
else:
    y = 3 * np.sin(2 * np.pi * ts / period)
data = 10 + y + np.random.normal(0, 0.3, y.shape)

# autocorrelation
ac, lags = autocrosscorr(data, stationary=True)
plot_replicate_array(
    ac,
    t=lags * np.median(np.diff(t)) / period,
    xlabel="lag in periods",
    ylabel="correlation",
)
