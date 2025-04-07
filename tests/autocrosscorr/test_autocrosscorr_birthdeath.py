import numpy as np
import matplotlib.pyplot as plt
import stochpy
from wela.autocrosscorr import autocrosscorr

smod = stochpy.SSA(model_file="birth_death.psc", dir=".")

# birth rate
k0 = 5
# death rate
d0 = 0.05
# final time of simulation
tf = 1500

# load the stochastic birth-death model
smod.ChangeParameter("k", k0)
smod.ChangeParameter("d", d0)

if False:
    # simulate stochastically
    smod.DoStochSim(end=tf, mode="time", trajectories=3, quiet=False)
    # plot the results
    smod.PlotSpeciesTimeSeries()
    plt.xlim([0, tf])
    plt.show()

# run many simulations
smod.DoStochSim(end=tf, mode="time", trajectories=500, quiet=False)

# put the trajectories on a grid - a matrix - with regularly spaced time points
ns = 1000
dt = np.mean(np.diff(np.linspace(0, tf, ns)))
timesavailable = np.linspace(0, tf, ns)
smod.GetRegularGrid(n_samples=ns)
# each row is one trajectory
data = np.array(smod.data_stochsim_grid.species[0]).astype("float")

# autocorrelation
ac, lags = autocrosscorr(data[:, int(ns / 2) :])
plt.figure()
t = dt * np.arange(int(ns / 2))
plt.plot(dt * lags, np.mean(ac, axis=0), "b-")
plt.plot(t, np.exp(-t * d0), "k--")
plt.grid()
plt.show()
