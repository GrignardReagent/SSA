# #!/usr/bin/env python3
# """
# Simulate a batch of Sobol sequence samples for transfer learning purposes, using the julia pipeline. 
# """

# import os
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import time
# from scipy.stats import qmc
# from simulation.mean_cv_t_ac import find_tilda_parameters
# from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
# from stats.mean import calculate_mean
# from stats.variance import calculate_variance
# from stats.cv import calculate_cv
# from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d
# import traceback
# from tqdm import tqdm

# np.random.seed(42)  # for reproducibility

# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------
# data_dir = "data"
# os.makedirs(data_dir, exist_ok=True)

# N = 1024
# sobol = qmc.Sobol(d=3, scramble=True, seed=42)
# U = sobol.random_base2(int(np.ceil(np.log2(N))))[:N]

# # Map Sobol sequence to parameter ranges
# mu_target   = qmc.scale(U[:, 0], 1, 10_000)
# cv_target   = qmc.scale(U[:, 1], 0.5, 2.0)
# t_ac_target = qmc.scale(U[:, 2], 5.0, 120.0)  # minutes

# max_runtime = 15 * 60 # seconds

# # -----------------------------------------------------------------------------
# # Simulation loop
# # -----------------------------------------------------------------------------
# success_count = 0
# failure_count = 0
# total_combinations = len(mu_target) 
# all_records = []

# print(f"Testing {total_combinations} parameter combinations...")
# print(f"Started at: {datetime.now()}")

# for combination_idx, (mu, t_ac, cv) in enumerate(
#     tqdm(zip(mu_target, t_ac_target, cv_target), total=total_combinations, desc="Simulating Sobol samples"),
#     start=1,
# ):
#     result_record = {
#         "mu_target": mu,
#         "t_ac_target": t_ac,
#         "cv_target": cv,
#         "success": False,
#         "error_message": "",
#         "rho": np.nan,
#         "d": np.nan,
#         "sigma_b": np.nan,
#         "sigma_u": np.nan,
#         "mu_observed": np.nan,
#         "cv_observed": np.nan,
#         "t_ac_observed": np.nan,
#         "variance_observed": np.nan,
#         "mean_rel_error_pct": np.nan,
#         "cv_rel_error_pct": np.nan,
#         "t_ac_rel_error_pct": np.nan,
#         "trajectory_filename": "",
#     }

#     try:
#         rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
#         result_record.update({"rho": rho, "d": d, "sigma_b": sigma_b, "sigma_u": sigma_u})

#         print(f"Testing mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f}")
#         print(f"  Parameters: rho={rho:.4f}, d={d:.4f}, sigma_b={sigma_b:.4f}, sigma_u={sigma_u:.4f}")

#         parameter_set = [
#             {"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}
#         ]

#         time_points = np.arange(0, 1_000, 1.0)
#         size = 500
#         # time the simulation 
#         start_time = time.time()
#         df_results = simulate_telegraph_model(parameter_set, time_points, size)
#         if time.time() - start_time > max_runtime:
#             raise RuntimeError(
#                 f"simulate_telegraph_model exceeded the runtime limit of {max_runtime} s."
#             )

#         trajectories = df_results[df_results["label"] == 0].drop("label", axis=1).values
        
#         # stat calculations
#         mean_observed = calculate_mean(trajectories, parameter_set, use_steady_state=True)
#         variance_observed = calculate_variance(trajectories, parameter_set, use_steady_state=True)
#         cv_observed = calculate_cv(variance_observed, mean_observed)

#         autocorr_results = calculate_autocorrelation(df_results)
#         ac_mean = autocorr_results["stress_ac"].mean(axis=0)
#         lags = autocorr_results["stress_lags"]
#         ac_time_observed = calculate_ac_time_interp1d(ac_mean, lags)

#         mean_rel_error = abs(mean_observed - mu) / mu
#         cv_rel_error = abs(cv_observed - cv) / cv
#         t_ac_rel_error = abs(ac_time_observed - t_ac) / t_ac
#         # end of calculations

#         result_record.update(
#             {
#                 "success": True,
#                 "mu_observed": mean_observed,
#                 "cv_observed": cv_observed,
#                 "t_ac_observed": ac_time_observed,
#                 "variance_observed": variance_observed,
#                 "mean_rel_error_pct": mean_rel_error * 100,
#                 "cv_rel_error_pct": cv_rel_error * 100,
#                 "t_ac_rel_error_pct": t_ac_rel_error * 100,
#             }
#         )

#         trajectory_filename = f"mRNA_trajectories_{mu:.3f}_{cv:.3f}_{t_ac:.3f}.csv"
#         trajectory_path = os.path.join(data_dir, trajectory_filename)
#         trajectory_df = pd.DataFrame(trajectories, columns=[f"t_{i}" for i in range(len(time_points))])
#         trajectory_df.to_csv(trajectory_path, index=False)
#         result_record["trajectory_filename"] = trajectory_filename

#         print(f"  Target: mu={mu:.3f}, cv={cv:.3f}, t_ac={t_ac:.3f}")
#         print(
#             f"  Observed: mu={mean_observed:.3f}, cv={cv_observed:.3f}, t_ac={ac_time_observed:.3f}"
#         )
#         print(
#             f"  Errors: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
#         )

#         if mean_rel_error < 0.2 and cv_rel_error < 0.2 and t_ac_rel_error < 0.2:
#             print("  ✅ All assertions passed")
#             success_count += 1
#         else:
#             print("  ❌ Tolerance check failed")
#             result_record["error_message"] = (
#                 f"Tolerance exceeded: mean={mean_rel_error:.1%}, cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
#             )
#             failure_count += 1

#     except Exception as e:
#         failure_count += 1
#         error_msg = str(e)
#         result_record["error_message"] = error_msg
#         print(f"  FAILED: mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f} - Error: {error_msg}")

#     all_records.append(result_record)
#     results_df = pd.DataFrame([result_record])
#     results_path = os.path.join(data_dir, "IY010_simulation_parameters_7_t_ac.csv")
#     write_header = not os.path.exists(results_path)
#     results_df.to_csv(results_path, mode="a", header=write_header, index=False)

# # -----------------------------------------------------------------------------
# # Summary
# # -----------------------------------------------------------------------------
# print("\n=== Final Results ===")
# print(f"Total combinations tested: {total_combinations}")
# print(f"Successful runs: {success_count}")
# print(f"Failed runs: {failure_count}")

# ## --
#!/usr/bin/env python3
"""
Simulate a Sobol-sampled synthetic dataset for transfer learning with domain randomisation (via introduced artefacts).
- Targets: (mu, CV, t_ac)
- Parameter solve: find_tilda_parameters()
- Dynamics: simulate_telegraph_model() via Julia
- Artefacts: missingness, bleaching/drift, heteroscedastic noise, spikes, scaling, mild time-warp
- Exports: per-parameter CSV (as before) + NPZ shard with metadata for DL pre-training
"""

import os
import json
import time
import math
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm

# --- Your existing imports ---
from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
from stats.mean import calculate_mean
from stats.variance import calculate_variance
from stats.cv import calculate_cv
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
GLOBAL_SEED = 42
rng = np.random.default_rng(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Sobol coverage
# Sobol coverage
N = 1024
sobol = qmc.Sobol(d=3, scramble=True, seed=GLOBAL_SEED)
U = sobol.random_base2(int(np.ceil(np.log2(N))))[:N]   # shape (N, 3)

# Scale all three dimensions in one call
l_bounds = np.array([1.0,   0.5,  5.0])     # [mu_min, cv_min, t_ac_min]
u_bounds = np.array([10_000.0, 2.0, 120.0]) # [mu_max, cv_max, t_ac_max]
targets = qmc.scale(U, l_bounds, u_bounds)  # shape (N, 3)

mu_target   = targets[:, 0]
cv_target   = targets[:, 1]
t_ac_target = targets[:, 2]

# Simulation
max_runtime = 15 * 60 # seconds per combination
trajectories_per_param = 500  # per parameter set
time_points = np.arange(0, 3000, 1.0)

# Export: also write shard-style NPZ every K parameter sets for easy DL loading
SHARD_SIZE = 16   # number of parameter-sets per shard (each set has 'trajectories_per_param' sequences)
shard_buffer = {
    "M": [],           # [num_sequences, T] augmented
    "M_clean": [],     # optional clean copy (same shapes)
    "t": [],           # list of time vectors (1 per set; may vary slightly with t_ac)
    "label": [],       # here just 0 (telegraph), leave room for future archetypes
    "meta": []         # per-sequence metadata dicts
}
shard_idx = 0

# Results log CSV
results_path = os.path.join(data_dir, "IY011_simulation_parameters_sobol.csv")
# start fresh if exists
if os.path.exists(results_path):
    os.remove(results_path)

# ---------------------------------------------------------------------
# Domain randomisation (artefacts)
# ---------------------------------------------------------------------
def apply_missingness(x, p_gap=0.15, max_gap=8, p_point=0.10):
    """Random contiguous gaps (NaNs) and scattered single-point NaNs."""
    y = x.copy()
    T = y.shape[-1]
    # scattered single NaNs
    if p_point > 0:
        mask = (rng.random(T) < p_point)
        y[mask] = np.nan
    # 0–2 contiguous gaps
    n_gaps = rng.integers(0, 3)
    for _ in range(n_gaps):
        if rng.random() < p_gap:
            g = rng.integers(1, max_gap + 1)
            s = rng.integers(0, max(1, T - g))
            y[s:s+g] = np.nan
    return y

def apply_bleach_and_drift(x, t, k_bleach_range=(0.0, 0.01), drift_amp_range=(-0.05, 0.05)):
    """Multiplicative bleaching exp(-k*t_norm) and additive slow drift (linear)."""
    y = x.copy().astype(float)
    t_norm = (t - t.min()) / max(1e-9, (t.max() - t.min()))
    k = rng.uniform(*k_bleach_range)
    mult = np.exp(-k * t_norm)
    drift_amp = rng.uniform(*drift_amp_range)
    drift = drift_amp * (t_norm - 0.5)
    y = y * mult + drift * np.nanmedian(y)
    return y

def apply_noise_and_spikes(x, base_sigma_frac=(0.01, 0.10), spike_prob=0.01, spike_mult=(2.0, 6.0)):
    """Heteroscedastic Gaussian noise + occasional spikes."""
    y = x.copy().astype(float)
    sigma_frac = rng.uniform(*base_sigma_frac)
    sigma = sigma_frac * max(1e-9, np.nanstd(y) or np.nanmean(np.abs(y)))
    noise = rng.normal(0.0, sigma, size=y.shape)
    y = y + noise
    # spikes
    if spike_prob > 0:
        mask = rng.random(y.shape[-1]) < spike_prob
        if mask.any():
            mult = rng.uniform(*spike_mult, size=mask.sum())
            y[mask] = y[mask] * mult
    return y

def apply_scaling(x, scale_range=(0.5, 2.0)):
    """Per-sequence rescaling to mimic field-of-view intensity changes."""
    s = rng.uniform(*scale_range)
    return x * s

def apply_time_warp(x, strength=0.05):
    """
    Mild non-uniform resampling by jittering indices and linear interpolation.
    strength is the max relative jitter.
    """
    y = x.copy().astype(float)
    T = y.shape[-1]
    base = np.arange(T, dtype=float)
    jitter = rng.uniform(-strength, strength, size=T)
    warped = np.clip(base + jitter * T, 0, T - 1)
    # linear interp onto base grid
    i0 = np.floor(warped).astype(int)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = warped - i0
    z = (1 - w) * y[i0] + w * y[i1]
    return z

def impute_nans_linear(x):
    """Simple linear interpolation over NaNs; leaves leading/trailing NaNs as nearest non-NaN."""
    y = x.copy().astype(float)
    n = y.shape[-1]
    idx = np.arange(n)
    mask = ~np.isnan(y)
    if mask.any():
        y_interp = np.interp(idx, idx[mask], y[mask])
        # carry ends
        if not mask[0]:
            first = np.argmax(mask)
            y_interp[:first] = y_interp[first]
        if not mask[-1]:
            last = n - 1 - np.argmax(mask[::-1])
            y_interp[last:] = y_interp[last]
        return y_interp
    else:
        return np.zeros_like(y)

def augment_sequence(x, t):
    """Compose artefacts with randomised application; returns augmented and a dict of what happened."""
    meta = {}
    y = x.astype(float)

    if rng.random() < 0.9:
        y2 = apply_missingness(y, p_gap=0.15, max_gap=8, p_point=0.10)
        meta["missingness"] = True
    else:
        y2 = y
        meta["missingness"] = False

    if rng.random() < 0.8:
        y3 = apply_bleach_and_drift(y2, t, k_bleach_range=(0.0, 0.01), drift_amp_range=(-0.05, 0.05))
        meta["bleach_drift"] = True
    else:
        y3 = y2
        meta["bleach_drift"] = False

    if rng.random() < 0.95:
        y4 = apply_noise_and_spikes(y3, base_sigma_frac=(0.01, 0.08), spike_prob=0.01, spike_mult=(2.0, 5.0))
        meta["noise_spikes"] = True
    else:
        y4 = y3
        meta["noise_spikes"] = False

    if rng.random() < 0.8:
        y5 = apply_scaling(y4, scale_range=(0.6, 1.6))
        meta["scaling"] = True
    else:
        y5 = y4
        meta["scaling"] = False

    if rng.random() < 0.5:
        y6 = apply_time_warp(y5, strength=0.05)
        meta["time_warp"] = True
    else:
        y6 = y5
        meta["time_warp"] = False

    # impute NaNs so DL doesn't choke; keep a flag
    if np.isnan(y6).any():
        y7 = impute_nans_linear(y6)
        meta["imputed"] = "linear"
    else:
        y7 = y6
        meta["imputed"] = "none"

    return y7.astype(np.float32), meta

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
success_count = 0
failure_count = 0
total_combinations = len(mu_target)

print(f"Testing {total_combinations} parameter combinations…")
print(f"Started at: {datetime.now()}")

for combo_idx, (mu, t_ac, cv) in enumerate(
    tqdm(zip(mu_target, t_ac_target, cv_target),
         total=total_combinations, desc="Simulating Sobol samples"), start=1
):
    # per-combination results row (clean fit metrics)
    record = {
        "mu_target": mu,
        "t_ac_target": t_ac,
        "cv_target": cv,
        "success": False,
        "error_message": "",
        "rho": np.nan,
        "d": np.nan,
        "sigma_b": np.nan,
        "sigma_u": np.nan,
        "mu_observed": np.nan,
        "cv_observed": np.nan,
        "t_ac_observed": np.nan,
        "variance_observed": np.nan,
        "mean_rel_error_pct": np.nan,
        "cv_rel_error_pct": np.nan,
        "t_ac_rel_error_pct": np.nan,
        "trajectory_filename": "",
        "seed": int(rng.integers(0, 2**31 - 1))
    }

    try:
        # --- Solve parameters for targets ---
        rho, d, sigma_b, sigma_u = find_tilda_parameters(mu, t_ac, cv)
        record.update({"rho": rho, "d": d, "sigma_b": sigma_b, "sigma_u": sigma_u})

        # --- Simulate (clean) ---
        parameter_set = [{"sigma_b": sigma_b, "sigma_u": sigma_u, "rho": rho, "d": d, "label": 0}]
        start_time = time.time()
        df_results = simulate_telegraph_model(parameter_set, time_points, trajectories_per_param)
        if (time.time() - start_time) > max_runtime:
            raise RuntimeError(f"simulate_telegraph_model exceeded {max_runtime} s")

        # Extract clean trajectories for label 0
        clean = df_results[df_results["label"] == 0].drop("label", axis=1).values  # [num_seq, T]

        # --- Fit metrics on clean (steady-state) ---
        mean_observed = calculate_mean(clean, parameter_set, use_steady_state=True)
        variance_observed = calculate_variance(clean, parameter_set, use_steady_state=True)
        cv_observed = calculate_cv(variance_observed, mean_observed)
        ac = calculate_autocorrelation(df_results)
        ac_mean = ac["stress_ac"].mean(axis=0)
        lags = ac["stress_lags"]
        ac_time_observed = calculate_ac_time_interp1d(ac_mean, lags)

        mean_rel_error = abs(mean_observed - mu) / max(1e-9, mu)
        cv_rel_error = abs(cv_observed - cv) / max(1e-9, cv)
        t_ac_rel_error = abs(ac_time_observed - t_ac) / max(1e-9, t_ac)

        record.update({
            "success": True,
            "mu_observed": mean_observed,
            "cv_observed": cv_observed,
            "t_ac_observed": ac_time_observed,
            "variance_observed": variance_observed,
            "mean_rel_error_pct": mean_rel_error * 100,
            "cv_rel_error_pct": cv_rel_error * 100,
            "t_ac_rel_error_pct": t_ac_rel_error * 100,
        })

        # --- Per-set CSV of clean sequences (as before) ---
        trajectory_filename = f"mRNA_trajectories_mu{mu:.3f}_cv{cv:.3f}_tac{t_ac:.3f}.csv"
        trajectory_path = os.path.join(data_dir, trajectory_filename)
        pd.DataFrame(clean, columns=[f"t_{i}" for i in range(len(time_points))]).to_csv(
            trajectory_path, index=False
        )
        record["trajectory_filename"] = trajectory_filename

        # --- Artefacts for DL (domain randomisation) ---
        aug_list = []
        meta_list = []
        for i in range(clean.shape[0]):
            y_aug, meta_aug = augment_sequence(clean[i], time_points)
            # Store normalised variant if you prefer (commented):
            # y_aug = (y_aug - np.nanmedian(y_aug)) / (np.nanmedian(np.abs(y_aug - np.nanmedian(y_aug))) + 1e-9)
            aug_list.append(y_aug.astype(np.float32))
            meta_aug.update({
                "mu_target": float(mu),
                "cv_target": float(cv),
                "t_ac_target": float(t_ac),
                "rho": float(rho), "d": float(d),
                "sigma_b": float(sigma_b), "sigma_u": float(sigma_u),
                "seed": int(record["seed"]),
                "label": 0,
                "T": int(len(time_points)),
                "combo_idx": int(combo_idx)
            })
            meta_list.append(meta_aug)

        # --- Add to shard buffer ---
        shard_buffer["M"].append(np.vstack(aug_list))
        shard_buffer["M_clean"].append(clean.astype(np.float32))
        shard_buffer["t"].append(time_points.astype(np.float32))
        shard_buffer["label"].append(np.zeros(clean.shape[0], dtype=np.int32))
        shard_buffer["meta"].extend(meta_list)

        # --- Flush shard if full ---
        if (combo_idx % SHARD_SIZE) == 0:
            shard_idx += 1
            shard_M = np.vstack(shard_buffer["M"])            # [S, T] (S = SHARD_SIZE*trajectories_per_param)
            shard_Mc = np.vstack(shard_buffer["M_clean"])
            # Time vectors may differ in length across rows; store separately per set
            shard_t_list = shard_buffer["t"]
            shard_labels = np.concatenate(shard_buffer["label"], axis=0)
            shard_meta = shard_buffer["meta"]

            shard_npz = os.path.join(data_dir, f"IY011_synth_shard_{shard_idx:03d}.npz")
            # Save as ragged by serialising t as object array of arrays
            np.savez_compressed(
                shard_npz,
                M=shard_M,
                M_clean=shard_Mc,
                label=shard_labels,
                # store lengths and concatenated t to avoid object arrays in NumPy
                t_concat=np.concatenate(shard_t_list),
                t_lengths=np.array([len(tt) for tt in shard_t_list], dtype=np.int32),
            )
            # Sidecar metadata JSON (list of dicts, one per sequence)
            with open(shard_npz.replace(".npz", ".json"), "w") as f:
                json.dump(shard_meta, f)

            # reset buffer
            shard_buffer = {"M": [], "M_clean": [], "t": [], "label": [], "meta": []}

        # --- Tolerance log ---
        if (mean_rel_error < 0.2) and (cv_rel_error < 0.2) and (t_ac_rel_error < 0.2):
            success_count += 1
        else:
            record["error_message"] = (
                f"Tolerance exceeded: mean={mean_rel_error:.1%}, "
                f"cv={cv_rel_error:.1%}, ac={t_ac_rel_error:.1%}"
            )
            failure_count += 1

    except Exception as e:
        failure_count += 1
        record["error_message"] = str(e)
        print(f"  FAILED: mu={mu:.2f}, t_ac={t_ac:.2f}, cv={cv:.2f} - Error: {e}")

    # append to rolling CSV
    pd.DataFrame([record]).to_csv(results_path, mode="a",
                                  header=not os.path.exists(results_path), index=False)

# ---------------------------------------------------------------------
# Flush any remaining shard
# ---------------------------------------------------------------------
if shard_buffer["M"]:
    shard_idx += 1
    shard_M = np.vstack(shard_buffer["M"])
    shard_Mc = np.vstack(shard_buffer["M_clean"])
    shard_t_list = shard_buffer["t"]
    shard_labels = np.concatenate(shard_buffer["label"], axis=0)
    shard_meta = shard_buffer["meta"]

    shard_npz = os.path.join(data_dir, f"IY011_synth_shard_{shard_idx:03d}.npz")
    np.savez_compressed(
        shard_npz,
        M=shard_M,
        M_clean=shard_Mc,
        label=shard_labels,
        t_concat=np.concatenate(shard_t_list),
        t_lengths=np.array([len(tt) for tt in shard_t_list], dtype=np.int32),
    )
    with open(shard_npz.replace(".npz", ".json"), "w") as f:
        json.dump(shard_meta, f)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n=== Final Results ===")
print(f"Total combinations tested: {total_combinations}")
print(f"Successful runs (within 20% on all targets): {success_count}")
print(f"Failed runs: {failure_count}")
print(f"Wrote results CSV: {results_path}")
print(f"Shards written: {shard_idx}")

