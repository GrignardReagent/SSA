import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import (
    calculate_autocorrelation,
    calculate_ac_time_interp1d,
)


CSV_PATH = "/home/ianyang/stochastic_simulations/experiments/EXP-25-IY010/IY010_simulation_parameters.csv"


def load_high_error_cases(csv_path: str) -> tuple[float, float]:
    """Return (mu, t_ac) for a case with large CV relative error."""
    df = pd.read_csv(csv_path)
    mask = (
        df["success"].fillna(False)
        & (df["cv_target"] >= 3.0) # take cases with high CV targets
        & (df["ac_rel_error_pct"].fillna(0) > 20.0) # pick cases with ac high relative errors (since that's where most variability was observed)
    )
    cases = df.loc[mask, ["mu_target", "t_ac_target", "cv_target"]].drop_duplicates()
    if cases.empty:
        raise ValueError("No high-error cases found in parameter file")
    return cases


def simulate_case(mu: float,  cv: float, t_ac: float, sizes: list[int]) -> pd.DataFrame:
    """Simulate multiple sample sizes and CVs, returning relative CV errors."""

    results = []
    time_points = np.arange(0, 
                            int(t_ac * 20),
                            1.0)
    # get the parameters for the given mu, cv, t_ac
    rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
    params = [{"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}]
    for size in sizes:
        print(f"Simulating case: mu={mu}, cv={cv}, t_ac={t_ac}, size={size}")
        df_results = simulate_one_telegraph_model_system(params, time_points, size)
        # drop the labels as we dont need them for analysis
        trajectories = df_results[df_results["label"] == 0].drop("label", axis=1).values
        # calculate the measured stats
        mean_obs = calculate_mean(trajectories, params, use_steady_state=True)
        var_obs = calculate_variance(trajectories, params, use_steady_state=True)
        cv_obs = calculate_cv(var_obs, mean_obs)
        ac = calculate_autocorrelation(df_results)
        ac_mean = ac["stress_ac"].mean(axis=0)
        lags = ac["stress_lags"]
        t_ac_obs = calculate_ac_time_interp1d(ac_mean, lags)
        
        # calculate the relative errors
        mean_rel_error = abs(mean_obs - mu) / mu
        cv_rel_error = abs(cv_obs - cv) / cv
        ac_rel_error = abs(t_ac_obs - t_ac) / t_ac

        results.append({
            "cv_target": cv, 
            "size": size,
            'mean_rel_error_pct': mean_rel_error * 100,
            'cv_rel_error_pct': cv_rel_error * 100,
            'ac_rel_error_pct': ac_rel_error * 100
                        })
    return pd.DataFrame(results)


cases = load_high_error_cases(CSV_PATH).head(5) # get the first 5 cases
# sweep sampling size (number of time series)
sizes = [100, 500, 1000, 5000, 10000]

all_results = []
# resimulate original cases
for _, row in cases.iterrows():
    mu, cv, t_ac = row["mu_target"], row["cv_target"], row["t_ac_target"]
    df_case = simulate_case(mu, cv, t_ac, sizes=sizes)
    # add the mu cv and t_ac to df_case
    df_case["mu cv t_ac"] = f"{mu:.3f}_{cv:.3f}_{t_ac:.3f}"
    
    # check if df_case is empty
    if df_case.empty:
        print(f"!!Warning!!: simulate_case returned empty DataFrame for mu={mu}, cv={cv}, t_ac={t_ac}")
        continue
    all_results.append(df_case)

if not all_results:
    raise ValueError("No valid simulation results obtained")
df = pd.concat(all_results, ignore_index=True)

# Create subplots for the three metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot CV relative error
cv_pivot = df.pivot_table(index="size", columns="mu cv t_ac", values="cv_rel_error_pct", aggfunc='mean')
cv_pivot.plot(marker="o", ax=axes[0])
axes[0].set_xlabel("Number of time series")
axes[0].set_ylabel("CV relative error (%)")
axes[0].set_title(f"CV relative error vs sample size")

# Plot mean relative error
mean_pivot = df.pivot_table(index="size", columns="mu cv t_ac", values="mean_rel_error_pct", aggfunc='mean')
mean_pivot.plot(marker="o", ax=axes[1])
axes[1].set_xlabel("Number of time series")
axes[1].set_ylabel("Mean relative error (%)")
axes[1].set_title(f"Mean relative error vs sample size")

# Plot autocorrelation relative error
ac_pivot = df.pivot_table(index="size", columns="mu cv t_ac", values="ac_rel_error_pct", aggfunc='mean')
ac_pivot.plot(marker="o", ax=axes[2])
axes[2].set_xlabel("Number of time series")
axes[2].set_ylabel("AC relative error (%)")
axes[2].set_title(f"AC relative error vs sample size")

# Add overall title with parameters
fig.suptitle(f"Relative errors (%) vs sample size")

plt.tight_layout()
# plt.show()
# Save the figure
plt.savefig("high_cv_sample_size_analysis.png")

