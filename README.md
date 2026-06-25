# Stochastic Simulations

This repository contains simulation, statistics, visualisation, and machine
learning code for stochastic gene expression time-series experiments. The
current simulation pipeline is centred on the two-state telegraph model:

- solve telegraph parameters from target summary statistics with
  `simulation.mean_cv_t_ac.find_tilda_parameters`
- simulate trajectories with the Julia-backed SSA wrapper
  `simulation.julia_simulate_telegraph_model.simulate_telegraph_model`
- analyse mean, variance, CV, autocorrelation, and steady-state behaviour
- train and evaluate downstream time-series classifiers and self-supervised
  models on synthetic and experimental data

Older pure-Python SSA simulators and exploratory parameter solvers have been
removed. New code should use the Julia wrapper and `find_tilda_parameters`.

## Repository Layout

```text
.
|-- experiments/          # One folder per experiment, e.g. EXP-26-IY025
|-- julia/                # Julia project and TelegraphSSA implementation
|-- notebooks/            # Exploratory notebooks and examples
|-- src/
|   |-- classifiers/      # SVM, random forest, logistic, neural classifiers
|   |-- dataloaders/      # Baseline, SSL, and SimCLR dataset loaders
|   |-- models/           # LSTM, transformer, SimCLR/SSL model components
|   |-- mutual_information/
|   |-- simulation/       # Current telegraph parameter solve + Julia wrapper
|   |-- stats/            # Mean, variance, CV, autocorrelation, reports
|   |-- training/         # Training/evaluation utilities
|   |-- utils/            # Data processing, labels, steady-state helpers
|   `-- visualisation/    # Trajectory, distribution, AC/cross-correlation plots
`-- pyproject.toml
```

## Environment

Use the `stochastic_sim` environment for normal development.

```bash
micromamba activate stochastic_sim
pip install -e .
```

If the environment needs to be recreated, install the Python dependencies from
`pyproject.toml` and keep the package editable with `pip install -e .`.

The simulator also uses the Julia project in `julia/`. The Python wrapper
activates and instantiates this Julia environment on first use through
`juliacall`.

## Simulating Telegraph Time Series

The current public simulation entry point is:

```python
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model
```

It expects a list of parameter dictionaries, an array of time points, and the
number of trajectories per parameter set. It returns a wide `pandas.DataFrame`
with a `label` column and one `time_*` column per time point.

```python
import numpy as np

from simulation.julia_simulate_telegraph_model import simulate_telegraph_model

parameter_sets = [
    {"sigma_b": 1.0, "sigma_u": 1.0, "rho": 10.0, "d": 1.0, "label": 0},
    {"sigma_b": 2.0, "sigma_u": 1.0, "rho": 15.0, "d": 1.0, "label": 1},
]

time_points = np.arange(0, 300, 1.0)
df = simulate_telegraph_model(parameter_sets, time_points, size=100, num_cores=4)
```

Julia thread count is fixed when Julia starts. If you need a specific thread
count, set it before the first `juliacall` import or first call to the wrapper:

```bash
export JULIA_NUM_THREADS=4
```

## Solving Parameters From Target Statistics

Use `find_tilda_parameters` to map target mean, autocorrelation time, and CV to
telegraph rates:

```python
import numpy as np

from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model

mu_target = 100.0
t_ac_target = 20.0
cv_target = 0.5

rho, d, sigma_b, sigma_u = find_tilda_parameters(
    mu_target,
    t_ac_target,
    cv_target,
    sigma_sum=5.0,
)

parameter_sets = [
    {
        "sigma_b": sigma_b,
        "sigma_u": sigma_u,
        "rho": rho,
        "d": d,
        "label": 0,
    }
]

time_points = np.arange(0, 1000, 1.0)
df = simulate_telegraph_model(parameter_sets, time_points, size=50)
```

`sigma_sum = sigma_b + sigma_u` controls the rescaling used by the solver. Some
target combinations are numerically ill-conditioned for the default
`sigma_sum=1.0`; for strict analytical checks, pass a problem-appropriate value
and tighten `max_rel_err`.

## Working With Simulator Output

Most statistics functions accept trajectory arrays with shape
`(n_trajectories, n_timepoints)`. For a simulator result:

```python
trajectories = df[df["label"] == 0].drop(columns=["label"]).to_numpy()
```

Autocorrelation utilities accept the full labelled DataFrame:

```python
from stats.autocorrelation import calculate_autocorrelation, calculate_ac_time_interp1d

ac = calculate_autocorrelation(df, stationary=True)
ac_mean = ac["stress_ac"].mean(axis=0)
t_ac_observed = calculate_ac_time_interp1d(ac_mean, ac["stress_lags"])
```

Plotting helpers in `visualisation.plots` accept either full simulator output
DataFrames or already-extracted trajectory matrices.

## Machine Learning Workflows

The repository contains several ML paths for synthetic and experimental time
series:

- `src/classifiers/`: classical and neural classifiers, including SVM,
  random forest, logistic regression, MLP, LSTM, and transformer interfaces
- `src/dataloaders/`: loaders for baseline supervised training and SimCLR-style
  self-supervised learning; `simclr.py` supports lazy loading plus instance,
  global, joint, and batch-wise normalisation modes
- `src/models/`: model definitions for LSTM, transformer, SimCLR/SSL, and
  related experiments
- `src/training/`: reusable training and evaluation utilities, including the
  cross-view InfoNCE loss used by recent SimCLR experiments

Recent experiments from `EXP-25-IY011` onward focus on SimCLR pretraining on
synthetic trajectories followed by downstream SVM classification on
experimental data.

Feature-engineering benchmarks such as catch22, tsfresh, and permutation or
shuffle controls currently live mainly in experiment folders. Promote reusable
pieces into `src/` only when they are needed across experiments.

## Experiments

Experiment folders live under `experiments/` and use names like
`EXP-26-IY019`, where `26` is the year and `IY019` is the experiment ID. A
typical experiment folder contains:

- a main Python script
- an `.out` log from the Python run
- optional Grid Engine `.sh`, `.o`, and `.e` files for Eddie/HPC jobs
- figures, CSV summaries, model checkpoints, or notebooks produced by the run

When creating a new experiment, also create the matching Electronic Lab
Notebook entry with the same experiment name.

The table below is a quick local index. The ELN should remain the source of
record for motivation, methods, results, and conclusions.

| Experiment | Brief description |
| --- | --- |
| `EXP-25-IY001` | Early LSTM architecture and hyperparameter selection on steady-state simulated trajectories, with many HPC job variants. |
| `EXP-25-IY002` | Follow-up LSTM training on concatenated steady-state simulation datasets, producing the `IY002A` model artifact. |
| `EXP-25-IY003` | Empty local placeholder folder; no experiment artifacts are currently present in the repository. |
| `EXP-25-IY004` | Synthetic telegraph-model classifier benchmarks across variance-ratio regimes. |
| `EXP-25-IY005` | Telegraph SSA fine-tuning and Fano-factor validation for the two-state model implementation. |
| `EXP-25-IY006` | Transformer architecture and hyperparameter-importance analysis, including trajectory-combination utilities and saved model/results artifacts. |
| `EXP-25-IY007` | Parameter-finding and classifier benchmarks across CV, `mu`, and `t_ac` variation settings, including LSTM/transformer sweeps and CV/autocorrelation diagnostics. |
| `EXP-25-IY008` | Experimental time-series processing for old and NEW TF-screen datasets: metadata mapping, full/post-switch extraction, CV analysis, and GFP/mCherry transforms. |
| `EXP-25-IY010` | Development and validation of telegraph target-statistic solvers, sampling strategies, synthetic dataset generation, measurement-noise tests, and ratio-classification benchmarks. |
| `EXP-25-IY011` | Sobol-sampled Julia simulations for baseline, `mu`, `cv`, and `t_ac` variation datasets, with baseline transformer, Siamese transformer, and initial contrastive-learning workflows. |
| `EXP-25-IY012` | Julia TelegraphSSA wrapper validation, fixed-statistic simulation tests, and analytical/histogram plotting checks. |
| `EXP-25-IY013` | Experimental time-series classification pipelines, including binary and multiclass SVM/LSTM/transformer comparisons and all-Omid pairwise heatmaps. |
| `EXP-26-IY014` | Julia-resimulated baseline, `mu`, `cv`, and `t_ac` variation datasets with 10-fold parameter differences, plus baseline transformer training and evaluation. |
| `EXP-26-IY015` | Baseline transformer experiments that augment raw simulated time series with additional summary-statistic inputs. |
| `EXP-26-IY016` | Follow-up SVM pipeline optimisation on IY014-style data, including 2-fold/10-fold variants and catch22/tsfresh feature benchmarks. |
| `EXP-26-IY017` | SimCLR follow-up from IY011 exploring batch size, embedding dimension, instance normalisation, catch22 baselines, and downstream SVM evaluation. |
| `EXP-26-IY018` | Baseline transformer hyperparameter sweeps with Optuna across baseline, `mu`, `cv`, and `t_ac` datasets, plus held-out test dataset generation. |
| `EXP-26-IY019` | Large Sobol synthetic dataset generation across baseline, `mu`, `cv`, and `t_ac` variation settings for downstream transformer training. |
| `EXP-26-IY020` | Additional synthetic dataset generation and summary-statistic processing across baseline, `mu`, `cv`, and `t_ac` variation settings. |
| `EXP-26-IY021` | Experimental-data benchmarks comparing raw SVM, catch22 + SVM, and frozen SimCLR embedding + SVM on mCherry, GFP, and dual-channel datasets. |
| `EXP-26-IY022` | SimCLR normalisation studies, including batch-wise, global, joint, and standard training variants with downstream SVM comparison. |
| `EXP-26-IY023` | Mixed-source SimCLR training across baseline, `mu`, `cv`, and `t_ac` variation datasets, with embedding visualisation and SVM downstream analysis. |
| `EXP-26-IY024` | Cross-view InfoNCE implementation note and SimCLR training comparisons for mixed variation datasets, including logit-matrix diagnostics. |
| `EXP-26-IY025` | Re-analysis of SVM pairwise performance, including median-split permutation tests, pairwise-variation controls, and `t_ac` mechanism studies linking fixed mean/CV sweeps to distribution shape, fraction-above-mean behaviour, and all four telegraph rates. |
| `EXP-26-IY026` | Fluorescence experiment inventory and overview, linking time-lapse files to strain/TF metadata and summary result tables. |
| `EXP-26-IY027` | Time-lapse dataset probe for Shimizu FRET-style `.mat` data, producing quick diagnostic plots of available trajectories. |
| `EXP-26-IY028` | Embedding-space and poster-figure analyses for experimental data, including SimCLR ROC-AUC, t-SNE/PHATE visualisations, clustering/discriminability metrics, and dual-channel catch22/SVM comparisons. |
| `EXP-26-IY029` | Pairwise same/different method comparison across catch22 + SVM, SimCLR + SVM, MLP, LSTM, and transformer models, with embedding visualisation and clustering metrics. |
| `EXP-26-IY030` | Supervised contrastive learning on dual-channel experimental data, including full SimCLR finetuning, frozen-backbone SupCon, and frozen-backbone MLP classifier variants. |
| `EXP-26-IY031` | TF identity/condition classification sanity checks on old and NEW experimental datasets, comparing full-trace and steady-state SimCLR results with balanced traces and confusion matrices. |
| `obsolete_files` | Archived prototypes and old simulation scripts; do not use as templates for new work. |

Historical experiments may reference removed APIs or absolute paths from the
machine or cluster where they were run. For new work, prefer the current `src/`
APIs above and keep experiment-specific paths local to the experiment folder.

## Notebooks

The root `notebooks/` directory contains older exploratory work, prototypes,
and WIP analyses. Recent reproducible work is usually easier to follow from the
matching `experiments/EXP-YY-IYXXX/` folder and ELN entry.

## Testing

Run tests from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest
```

Useful focused checks:

```bash
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest src/simulation/tests
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest src/visualisation/tests
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest src/stats/tests
```

The Julia-backed simulation tests may take longer on first run because Julia
has to initialise and instantiate the `julia/` project.

## Plotting Conventions

For experiment figures:

- save figures as `IYXXX_<figure_name>.png`
- label axes with variable names and units, for example `Time / min`
- include error bars for statistical plots, using SEM by default
- use `sns.color_palette("colorblind")` where possible
- prefer outside legends when in-axes legends cover data
- call `tight_layout()` or use `constrained_layout=True`

See `src/visualisation/plots.py` for reusable plotting helpers.

## HPC Jobs

Long simulations are often run on the University of Edinburgh Eddie cluster
with Grid Engine. Job scripts should activate `stochastic_sim`, request the
needed GPU/CPU resources, set a runtime limit, and write Python output to a
matching `.out` file in the experiment directory.

## Notes

- Use the Julia wrapper for new simulations.
- Do not reintroduce the deleted pure-Python telegraph SSA simulator.
- Keep experiment changes scoped to the relevant `EXP-YY-IYXXX` folder.
- Generated data, logs, checkpoints, and figures can be large; avoid committing
  bulky outputs unless they are deliberate experiment artefacts.
