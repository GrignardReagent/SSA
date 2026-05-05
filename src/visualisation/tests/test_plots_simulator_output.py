import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualisation.plots import (
    plot_autocorr,
    plot_mRNA_dist,
    plot_mRNA_trajectory,
    plot_mRNA_variance,
)


def _simulator_like_df():
    time_points = np.arange(0, 20, 1.0)
    columns = [f"time_{t}" for t in time_points]
    rows = []
    labels = []
    for label, offset in [(0, 2), (1, 5)]:
        for rep in range(3):
            rows.append(offset + rep + np.arange(len(time_points)) % 4)
            labels.append(label)

    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, "label", labels)
    return df, time_points


def _parameter_sets():
    return [
        {"sigma_b": 0.8, "sigma_u": 0.4, "rho": 20.0, "d": 10.0, "label": 0},
        {"sigma_b": 0.9, "sigma_u": 0.5, "rho": 25.0, "d": 10.0, "label": 1},
    ]


def test_trajectory_and_variance_accept_simulator_output():
    df, time_points = _simulator_like_df()
    params = _parameter_sets()

    fig, ax = plot_mRNA_trajectory(params, time_points, df)
    assert ax.get_xlabel() == "Time / min"
    assert ax.get_ylabel() == "mRNA count / molecules"
    plt.close(fig)

    fig, ax = plot_mRNA_variance(params, time_points, df)
    assert ax.get_xlabel() == "Time / min"
    assert ax.get_ylabel() == "Variance of mRNA count / molecules^2"
    plt.close(fig)


def test_distribution_and_autocorr_accept_simulator_output():
    df, _ = _simulator_like_df()
    params = _parameter_sets()

    fig, ax = plot_mRNA_dist(params, df, kde=False)
    assert ax.get_xlabel() == "mRNA count at steady state / molecules"
    plt.close(fig)

    fig, ax = plot_autocorr(params, df)
    assert ax.get_xlabel() == "Lag / time steps"
    plt.close(fig)
