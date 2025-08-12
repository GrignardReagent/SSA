"""Generate biologically valid (mu, t_ac, cv) target triples.

This module provides a small helper function and command line interface for
sampling target statistics that are guaranteed to produce valid kinetic
parameters when passed to :func:`simulation.mean_cv_t_ac.find_tilda_parameters`.

Running this script directly writes a CSV file containing 10,000 verified
target combinations which can then be fed into downstream simulations.
"""

import numpy as np
from simulation.mean_cv_t_ac import find_tilda_parameters


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
        Seed for the pseudo-random number generator to ensure reproducibility.

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
        # # Pick a mean expression level and derive admissible CV bounds
        # mu = rng.uniform(1.0, 2000.0)
        # # FF, or CV^2 * mu., has to be between 1 and 20. 
        # # CV has to be lower than 5.0, but since the highest CV value is sqrt(20/1.0) =4.47 anyways, there's no need to define this again. 
        # cv_low = np.sqrt(1.0 / mu)
        # cv_high = np.sqrt(20.0 / mu)
        
        #TODO: pick more diverse CV values

        # # Skip if the bounds are invalid for this ``mu``
        # if cv_low >= cv_high:
        #     continue

        # Sample coefficient of variation within the valid range
        # cv = rng.uniform(cv_low, cv_high)
        
        # Sample CV and Fano factor (FF) within the desired ranges.
        cv = rng.uniform(0.1, 5.0)
        ff = rng.uniform(1.0, 20.0)        
        # Derive the corresponding mean from FF = mu * cv^2 without
        # explicitly constraining ``mu``.
        mu = ff / (cv**2)
        
        # Sample a reasonable autocorrelation time
        t_ac = rng.uniform(0.5, 100.0)

        try:
            # Validate that parameters exist for this target triple
            find_tilda_parameters(mu, t_ac, cv, check_biological=True)
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
    np.savetxt(
        "/home/ianyang/stochastic_simulations/experiments/EXP-25-IY010/data/valid_targets.csv",
        combos,
        delimiter=",",
        header="mu,t_ac,cv",
        comments="",
    )
