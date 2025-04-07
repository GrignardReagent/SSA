import numpy as np
import matplotlib.pylab as plt
from mywela.autocrosscorr import autocrosscorr
from mywela.plotting import plot_replicate_array

nr = 1000
nt = 500
period = 8
noise_sig = 0.01
t = np.linspace(0, period * 4, nt)
ts = np.tile(t, nr).reshape((nr, nt))

final_t = 2 * np.pi * ts / period + np.pi / 4 * np.random.rand(nr, 1)
y = 3 * np.sin(final_t)
s_sin = 10 + y + np.random.normal(0, noise_sig, y.shape)
z = 3 * np.cos(final_t)
s_cos = 10 + z + np.random.normal(0, noise_sig, z.shape)

# correlation
cs, lags = autocrosscorr(s_cos, s_sin)
sc, lags = autocrosscorr(s_sin, s_cos)

# cosine is delayed by period/4 = pi/2 compared to sine
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t / period, np.sin(2 * np.pi * t / period), label="sine")
plt.plot(t / period, np.cos(2 * np.pi * t / period), label="cosine")
plt.legend()
plt.grid()
plt.xlim([0, 1])
plt.subplot(3, 1, 2)
# peaks at 0.25
plot_replicate_array(
    cs,
    t=lags * np.median(np.diff(t)) / period,
    ylabel="correlation",
    title="<cos * sin>",
    show=False,
)
plt.xlim([-1, 1])
plt.subplot(3, 1, 3)
# peaks at -0.25
plot_replicate_array(
    sc,
    t=lags * np.median(np.diff(t)) / period,
    xlabel="lag in periods",
    ylabel="correlation",
    title="<sin * cos>",
    show=False,
)
plt.xlim([-1, 1])
plt.tight_layout()
plt.show()
