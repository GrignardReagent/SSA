"""Run telegraph-model simulations in a Latin square design.

This script demonstrates how to leverage the precomputed grid of
(mu, t_ac, cv) target statistics in ``data/valid_targets.csv`` to generate
synthetic trajectories using a Latin square experimental design.  A Latin
square ensures that each coefficient of variation (cv) level appears exactly
once in each row and column of the design matrix, mitigating potential order
effects and blocking nuisance variation.

The actual trajectory saving line is left commented so users can inspect the
results before deciding where and how to store them.
"""

from pathlib import Path

import numpy as np
from simulation.mean_cv_autocorr_v2 import find_tilda_parameters
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system


def latin_square_simulation(num_levels: int = 3, seed: int = 0,
                            size: int = 100, time_end: float = 144.0) -> None:
    """Simulate combinations in a Latin square design.

    Parameters
    ----------
    num_levels: int
        Number of distinct values to draw for each statistic (mu, t_ac, cv).
    seed: int
        Seed for the pseudo-random number generator.
    size: int
        Number of stochastic trajectories to simulate for each condition.
    time_end: float
        End time (in minutes) for the simulation; time points are sampled every
        minute starting at zero.
    """
    rng = np.random.default_rng(seed)

    # Load the valid target grid produced by ``IY010_generate_valid_targets.py``
    targets_path = Path(__file__).resolve().parent / "data" / "valid_targets.csv"
    targets = np.loadtxt(targets_path, delimiter=",", skiprows=1)

    # Randomly choose ``num_levels`` unique values for each statistic
    mus = rng.choice(targets[:, 0], size=num_levels, replace=False)
    t_acs = rng.choice(targets[:, 1], size=num_levels, replace=False)
    cvs = rng.choice(targets[:, 2], size=num_levels, replace=False)

    # Build a Latin square: each row is a rotation of [0, 1, ..., num_levels-1]
    latin_square = np.array([np.roll(np.arange(num_levels), -i)
                             for i in range(num_levels)])

    time_points = np.arange(0.0, time_end, 1.0)

    # Iterate over the design matrix
    for i, mu in enumerate(mus):
        for j, t_ac in enumerate(t_acs):
            cv = cvs[latin_square[i, j]]

            # Convert target statistics to kinetic parameters
            rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
            params = [{
                "sigma_b": sigma_b,
                "sigma_u": sigma_u,
                "rho": rho,
                "d": d,
                "label": i * num_levels + j,
            }]

            # Run the simulation for this combination
            df_results = simulate_one_telegraph_model_system(
                params, time_points, size, num_cores=1
            )

            # Uncomment the line below to save trajectories for later analysis
            # df_results.to_csv(
            #     targets_path.parent / f"sim_mu{mu:.2f}_tac{t_ac:.2f}_cv{cv:.2f}.csv",
            #     index=False,
            # )

            print(f"Simulated mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f}")


if __name__ == "__main__":
    latin_square_simulation()
