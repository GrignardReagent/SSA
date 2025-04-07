from mywela.autocrosscorr import autocrosscorr
from mywela.plotting import plot_replicate_array
import pandas as pd

df = pd.read_csv("by4741_omero20016.csv")
v = df.drop("cellID", axis=1).to_numpy()
ac, lags = autocrosscorr(v)
plot_replicate_array(ac, t=lags * 5 / 60, xlabel="lag", ylabel="correlation")
