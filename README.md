# Stochastic Simulations

Simulation, statistical analysis, and machine-learning workflows for stochastic
gene-expression time series. The main model is the two-state telegraph model;
recent work uses synthetic trajectories for supervised and contrastive learning
and evaluates learned representations on experimental fluorescence data.

For new work, treat `src/` as the reusable library and each
`experiments/EXP-YY-IYXXX/` directory as an experiment record. Historical
notebooks and archived prototypes are described separately below.

## Quick Start

Use the repository's `stochastic_sim` environment:

```bash
micromamba env create -f requirements.yml
micromamba activate stochastic_sim
pip install -e .
```

The simulator also needs Julia and the Python `juliacall` bridge. On its first
call, the wrapper activates and instantiates the Julia project in `julia/`.
The feature and test workflows additionally use `pycatch22` and `pytest`.
These three packages are used by the code but are not currently declared in
the checked-in Python dependency files, so install them when needed:

```bash
pip install juliacall pycatch22 pytest
```

Imports assume the editable installation above. For one-off commands, use
`PYTHONPATH=src` from the repository root.

## Repository Layout

```text
.
|-- src/                  # Reusable Python library
|   |-- classifiers/      # Classical and neural classifier wrappers
|   |-- dataloaders/      # Supervised, pairwise, SSL, and SimCLR loaders
|   |-- features/         # catch22 extraction and benchmarks
|   |-- models/           # MLP, LSTM, transformer, and SSL models
|   |-- mutual_information/
|   |-- simulation/       # Telegraph parameter solver and Julia wrapper
|   |-- stats/            # Mean, variance, CV, Fano factor, and correlation
|   |-- training/         # Training, evaluation, contrastive losses, diagnostics
|   |-- utils/            # Processing, augmentation, embeddings, and metrics
|   `-- visualisation/    # Reusable trajectory and statistics plots
|-- experiments/          # One directory per experiment record
|-- julia/                # Julia environment and SSA implementations
|-- notebooks/            # Older exploratory notebooks and local artifacts
|-- requirements.yml      # Preferred micromamba environment
`-- pyproject.toml        # Editable Python package metadata
```

## Current Simulation Workflow

Use `find_tilda_parameters` to solve telegraph rates from a target mean,
autocorrelation time, and coefficient of variation. It returns parameters in
the order `(rho, d, sigma_b, sigma_u)`.

```python
import numpy as np

from simulation.mean_cv_t_ac import find_tilda_parameters
from simulation.julia_simulate_telegraph_model import simulate_telegraph_model

rho, d, sigma_b, sigma_u = find_tilda_parameters(
    mu_target=10.0,
    t_ac_target=2.0,
    cv_target=0.5,
    sigma_sum=5.0,
    max_rel_err=0.01,
)

parameter_sets = [
    {
        "sigma_b": sigma_b,
        "sigma_u": sigma_u,
        "rho": rho,
        "d": d,
        "label": "example",
    }
]
time_points = np.arange(0.0, 144.0, 1.0)
results = simulate_telegraph_model(
    parameter_sets,
    time_points,
    size=100,
    num_cores=4,
)
```

`simulate_telegraph_model` returns a wide `pandas.DataFrame`: one row per
trajectory, a `label` column, and one `time_*` column per requested time point.
Labels may be strings or numbers. To obtain a trajectory matrix:

```python
trajectories = results.drop(columns="label").to_numpy()
```

Julia's thread count is fixed when Julia starts. Set `JULIA_NUM_THREADS` before
the first `juliacall` import or simulator call when a fixed thread count is
required:

```bash
export JULIA_NUM_THREADS=4
```

The parameter solver requires positive targets and
`mu_target * cv_target**2 > 1`. `sigma_sum` is the fixed value of
`sigma_b + sigma_u`; it can affect numerical conditioning and is not selected
automatically.

## Reusable Analysis and ML Code

- `stats.autocorrelation` calculates auto/cross-correlation and interpolated
  autocorrelation time; the other `stats` modules provide analytical and
  trajectory-based summary statistics.
- `utils.processing` contains the shared preprocessing pipeline. Missing values
  are handled with deterministic `IterativeImputer` logic that preserves shape,
  with guarded fallbacks for pathological inputs.
- `dataloaders.simclr` supports synthetic and experimental contrastive pairs,
  lazy loading, and instance, global, joint, or batch-wise normalization.
- `training.train` contains supervised, Siamese, SimCLR cross-view InfoNCE, and
  supervised-contrastive training loops. Evaluation, few-shot utilities, and
  diagnostics live beside it.
- `features.catch22`, `utils.embeddings`, and `utils.metrics` support the common
  downstream SVM/k-NN and representation-analysis workflows.
- `visualisation.plots` accepts simulator-style DataFrames or extracted
  trajectory matrices for the main telegraph-model plots.

Before adding a helper to an experiment, check `src/` for an existing reusable
implementation.

## Experiment Index

Experiment directories are named `EXP-YY-IYXXX`, where `YY` is the year and
`IYXXX` is the experiment identifier. Scripts, job files, logs, figures,
checkpoints, and result tables in these directories preserve the local record;
the matching [Electronic Lab Notebook](https://www.notion.so/202698419cbd8055bfc9db5bbf88b149?v=202698419cbd8092be3d000cc75d14e6&source=copy_link)
entry is the source of record for motivation, methods, results, and conclusions.

| Experiment | Scope |
| --- | --- |
| `EXP-25-IY001` | LSTM architecture and hyperparameter selection on steady-state synthetic trajectories, with HPC variants. |
| `EXP-25-IY002` | Historical follow-up that assembles variance-ratio steady-state data and trains the selected LSTM. |
| `EXP-25-IY003` | Earlier LSTM architecture-selection and hyperparameter-finetuning runs preserved in two subdirectories. |
| `EXP-25-IY004` | Historical telegraph classifier benchmarks across variance-ratio regimes using the selected LSTM. |
| `EXP-25-IY005` | Telegraph SSA tuning and Fano-factor validation. |
| `EXP-25-IY006` | Transformer architecture and hyperparameter analysis, trajectory combination, and saved model artifacts. |
| `EXP-25-IY007` | Historical parameter-solver validation and LSTM/transformer benchmarks across target-statistic variations. |
| `EXP-25-IY008` | Processing and analysis of old and expanded TF-screen fluorescence time series. |
| `EXP-25-IY010` | Historical target-statistic solver development, synthetic sampling, noise tests, and ratio classification. |
| `EXP-25-IY011` | Sobol-sampled Julia simulations plus baseline, Siamese, and early contrastive transformer workflows. |
| `EXP-25-IY012` | Notebook-only Julia telegraph wrapper validation and analytical/distribution sanity checks. |
| `EXP-25-IY013` | Raw SVM, LSTM, and transformer classification of old/new experimental TF-condition traces. |
| `EXP-26-IY014` | Julia-resimulated baseline and isolated mean, CV, or autocorrelation-time variation datasets. |
| `EXP-26-IY015` | Baseline transformer studies of 2-fold parameter differences and trajectory-group size. |
| `EXP-26-IY016` | SVM optimization and catch22/tsfresh benchmarks on 2-fold and 10-fold synthetic variations. |
| `EXP-26-IY017` | SimCLR batch-size, embedding-width, and normalization studies with catch22 and downstream SVM baselines. |
| `EXP-26-IY018` | Optuna transformer sweeps across synthetic variation datasets and held-out test generation. |
| `EXP-26-IY019` | Large Sobol synthetic corpora for baseline and isolated mean, CV, or autocorrelation-time variations. |
| `EXP-26-IY020` | Denser synthetic corpora and summary-statistic/steady-state dataset preparation. |
| `EXP-26-IY021` | Raw SVM, catch22+SVM, and frozen SimCLR+SVM benchmarks on single- and dual-channel experimental data. |
| `EXP-26-IY022` | Instance, global, joint, and batch-wise normalization studies for SimCLR and downstream SVMs. |
| `EXP-26-IY023` | Mixed-variation SimCLR training across normalization, batch-size, sequence-length, and embedding settings. |
| `EXP-26-IY024` | Cross-view InfoNCE validation and comparison with standard InfoNCE on mixed synthetic data. |
| `EXP-26-IY025` | Pairwise SVM, catch22, and telegraph-mechanism analyses of autocorrelation-time variation. |
| `EXP-26-IY026` | OMERO fluorescence/TF dataset survey with deterministic metadata parsing and an LLM fallback. |
| `EXP-26-IY027` | Exploratory probes of Fraisse VRAE and Shimizu FRET time-lapse datasets. |
| `EXP-26-IY028` | SimCLR/catch22 representation, ROC-AUC, clustering, and poster analyses of experimental data. |
| `EXP-26-IY029` | Pairwise comparison of raw SVM, catch22, SimCLR, MLP, LSTM, and transformer methods. |
| `EXP-26-IY030` | MLP versus RBF-SVM heads on frozen SimCLR embeddings for TF-at-condition classification. |
| `EXP-26-IY031` | Full-trace and steady-state TF/condition classification and embedding-separation checks. |
| `EXP-26-IY032` | k-NN embedding retrieval and leave-one-TF-out open-set diagnostics. |
| `EXP-26-IY033` | Condition-first classification after collapsing TF identity into carbon-source classes. |
| `EXP-26-IY034` | Supervised-contrastive refinement of SimCLR backbones with frozen-embedding SVM evaluation. |
| `EXP-26-IY035` | SimCLR pretraining on experimental same-file, augmented, and full-versus-steady-state views. |
| `EXP-26-IY036` | All-class supervised-contrastive encoders and k-NN/SVM comparisons; v3 follow-up scripts have no recorded results. |

Create a matching ELN page when adding an experiment. `EXP-26-IY026` has its
own README and additional OMERO dependency notes.

## Tests

Run the test suite from the repository root:

```bash
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest
```

Focused suites are under `src/*/tests/`, for example:

```bash
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest src/simulation/tests
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest src/training/tests
PYTHONPATH=src micromamba run -n stochastic_sim python -m pytest src/utils/tests
```

Julia-backed tests can be slower on first use while the Julia project is
instantiated.

## Experiment Conventions

- Use the `stochastic_sim` environment for local work. Eddie Grid Engine jobs
  use `conda activate stochastic_sim` after loading Miniforge and CUDA modules.
- Export one `RUN_TIMESTAMP` before a job so logs, checkpoints, histories, and
  tracking runs receive the same timestamp.
- Save figures as `IYXXX_<figure_name>.png`; label axes with variable and units,
  use colorblind-safe colors, and include error bars (standard deviation by
  default, or state the alternative).
- Keep experiment-specific scripts and artifacts in their experiment directory;
  move code into `src/` only when it is reusable.
- Avoid committing large generated data, logs, or checkpoints unless they are
  deliberate experiment records.

## Historical and Archival Material

`experiments/obsolete_files/` contains archived prototypes and old simulation
code. Root `notebooks/` and the Julia notebooks are mostly exploratory or
historical. They may contain machine-specific paths or interfaces that no
longer represent the supported workflow; use the current APIs shown above for
new work.
