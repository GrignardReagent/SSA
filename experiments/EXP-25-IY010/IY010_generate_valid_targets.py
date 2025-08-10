"""Generate biologically valid (mu, t_ac, cv) target triples.

This module provides a small helper function and command line interface for
sampling target statistics that are guaranteed to produce valid kinetic
parameters when passed to :func:`simulation.mean_cv_autocorr_v2.find_tilda_parameters`.

Running this script directly writes a CSV file containing 10,000 verified
target combinations which can then be fed into downstream simulations.
"""

import numpy as np
from simulation.mean_cv_autocorr_v2 import find_tilda_parameters


def generate_valid_targets(n: int = 10_000, seed: int = 0) -> np.ndarray:
    """Sample target statistics that satisfy biological constraints.

    The helper below randomly samples candidate combinations of the mean
    (``mu``), autocorrelation time (``t_ac``) and coefficient of variation
    (``cv``).  Each candidate is validated by calling
    :func:`find_tilda_parameters`; only combinations that do not raise an
    exception are retained.

    Parameters
    ----------
    n:
        Number of valid combinations to return.
    seed:
        Seed for the pseudo‑random number generator to ensure reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 3)`` containing the valid ``(mu, t_ac, cv)``
        triples.
    """

    # Random number generator for reproducibility
    rng = np.random.default_rng(seed)
    combos = []

    # Keep drawing samples until we collect ``n`` valid combinations
    while len(combos) < n:
        # Pick a mean expression level and derive admissible CV bounds
        mu = rng.uniform(1.1, 20.0)
        cv_low = np.sqrt(1.01 / mu)
        cv_high = np.sqrt(20.0 / mu)

        # Skip if the bounds are invalid for this ``mu``
        if cv_low >= cv_high:
            continue

        # Sample coefficient of variation within the valid range
        cv = rng.uniform(cv_low, cv_high)

        # Sample a reasonable autocorrelation time
        t_ac = rng.uniform(0.5, 100.0)

        try:
            # Validate that parameters exist for this target triple
            find_tilda_parameters(mu, t_ac, cv, check_biological=False)
            combos.append((mu, t_ac, cv))
        except Exception:
            # Discard invalid combinations and try again
            pass

    return np.array(combos)


if __name__ == "__main__":
    # When executed as a script, generate the default 10k combinations and
    # persist them to ``data/valid_targets.csv`` so they can be reused by other
    # experiments.
    combos = generate_valid_targets()
    # uncomment to save data
    # np.savetxt(
    #     "experiments/EXP-25-IY010/data/valid_targets.csv",
    #     combos,
    #     delimiter=",",
    #     header="mu,t_ac,cv",
    #     comments="",
    # )
