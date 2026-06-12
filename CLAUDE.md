# General Guidelines

## 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

## 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

## 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

## 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting
- Avoid hardcoding values in scripts
- Provide sufficient in-line, step-by-step comments and docstrings for codes, functions and classes
- When writing code, descriptions (e.g., Jupyter notebooks or ELNs) or comments, be succinct and to the point, while ensuring clarity. Avoid unnecessary verbosity unless asked to, but do not sacrifice important details for brevity. Strive for a balance between conciseness and informativeness in all communications.

## 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

# Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

# Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

# Environment Setup

Always use the micromamba `stochastic_sim` environment unless explicitly stated otherwise.

# Common Commands

Use ``simulate_telegraph_model`` from ``simulation.julia_simulate_telegraph_model`` to simulate time series data from the telegraph model. 

# Data Preprocessing

## Missing Values

- Use `sklearn.impute.IterativeImputer` as the default method for handling NaNs in experimental time-series matrices.
- Always enable it with `from sklearn.experimental import enable_iterative_imputer  # noqa: F401` before importing `IterativeImputer`.
- Prefer a deterministic imputer configuration with `random_state=42` unless an experiment defines a different fixed seed.
- Do not use ad hoc row interpolation or median filling as the primary NaN strategy. Median/global fills are acceptable only as a guarded fallback for pathological cases such as all-NaN timepoints or residual NaNs after iterative imputation.
- Preserve the original matrix shape after imputation and verify that no NaNs remain.

# Architecture Overview

## Source Layout

**`experiments/`** — One folder per experiment run, named `EXP-YY-IYXXX`, where `YY` is the year and `IYXXX` is the experiment identifier, e.g., `EXP-26-IY023` for the experiment number IY023 created in 2026.

## Self-Supervised Learning (SimCLR)

`src/dataloaders/simclr.py` provides contrastive learning dataloaders. The cross-view InfoNCE loss lives in `src/training/`. Experiments EXP-25-IY011 onwards explore SimCLR pretraining on synthetic data followed by SVM downstream classification on experimental data.

## Experiment Structure

Each experiment folder typically contains:

- `.py` — main ML training, simulation or data analysis script
- `out` — output file for logs and results from the python script, usually with the same name but `.out` extension
- `.sh` — Grid Engine job script submitted to HPC, which calls the `.py` script, usually with the same name but `.sh` suffix. Sometimes this is simply a shell script for running several python scripts
- `.o` / `.e` — stdout / stderr logs from cluster runs

## HPC Job Submission

The codes are usually ran on a Linux machine with 1 GPU, but sometimes codes are submitted to an HPC cluster. To submit a job to the University of Edinburgh Eddie cluster (using Grid Engine).

Every job script must follow the following conventions listed in the example (`IYXXX_job_script.sh`):

```bash
#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Job name
#$ -N IYXXX_job_script
#$ -o IYXXX_job_script.o$JOB_ID
#$ -e IYXXX_job_script.e$JOB_ID

# Use the current working dir
#$ -cwd

# Max runtime limit (48h) - this must be listed, or the job won't run, max is 48h on Eddie.
#$ -l h_rt=47:59:59

# Request 1 GPUs in the gpu queue, or however many the user specifies
#$ -q gpu 
#$ -l gpu-mig=1

# Request 32G per core (32G × 4 cores virtual memory)
#$ -l h_rss=32G

# Email notifications on job begin/end/abort
#$ -m bea -M s1732775@ed.ac.uk 

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load miniforge
module load cuda

# Activate env (use conda instead of micromamba on Eddie)
conda activate stochastic_sim

# Run Python script and log output
python IYXXX_job_script.py > IYXXX_job_script.out 2>&1

# Deactivate after job is done
conda deactivate
```

# Plotting

- **File Output:** Always save figures as `IYXXX_<figure_name>.png`.
- **Labels:** Both axes must be labeled with the variable name **and units**, e.g. `"Time / min"`, `"mRNA count / molecules"`. Never leave an axis unlabeled.
- **Bar Labels:** Each bar must be labeled with its category name or value, depending on the context.
- **Legend:** Place in upper right or outside if covering data.
- **Consistency:** Use consistent colors for the same variable across different figures.
- **Scientific Notation:** Use `matplotlib.ticker` for axes when data is in large/small ranges.
- **Numerical Formatting:** Report all displayed values (tick labels, annotations, legend entries) to **2 significant figures**.
- **Figure Dimensions:**
  - Default figure size: (8, 5) for single plots
  - Wide/time-series data: (12, 4) or (14, 4)
  - Square plots (phase planes, heatmaps, scatter): (6, 6) or (7, 7)
  - Loss/training curves: (8, 4)
  - Multi-panel (1 row, N cols): (5*N, 4) per column
  - Multi-panel (M rows, 1 col): (8, 4*M) per row
  - Multi-panel grid (M×N): (5*N, 4*M)
  - For figure legends, always use `loc="best"` to avoid covering data.
  - Always use `tight_layout()` or `constrained_layout=True`
  - DPI: 150 for screen display, 300 for export/paper

## Shared Axes in Multi-Panel Figures

When multiple subplots are comparing the **same quantity** across conditions (e.g. time series from different classes, scatter plots of the same statistic), always share the relevant axis so panels are directly comparable:

- Use `sharey=True` in `plt.subplots(...)` when the y-axis represents the same variable across panels.
- Use `sharex=True` when the x-axis represents the same variable (e.g. time).
- Use both when appropriate (e.g. a grid of time series all on the same scale).
- Do **not** share axes when panels intentionally show different quantities with different natural ranges.

## General Styling & Aesthetics

- **Font**: Use "sans-serif" for all text.
- **Font Sizes**:
    - Titles: 14pt
    - Labels: 12pt
    - Ticks: 10pt
    - Legend: 10pt
- **Colours**: Use `sns.color_palette("colorblind")` as the default palette for all plots.

## Error Bars

Every plot of statistical data must include error bars. Use the standard deviation (std) by default, and when std isn't available, use the standard error of the mean (SEM); state explicitly in the caption or legend if a different measure is used (e.g. 95% CI).

## Text Overlap & Legend Placement

**Always verify** that no text elements collide before considering a figure done.

- **Bar value annotations:** Use `fontsize=8` for annotation labels on charts with more than 10 bars; they will overlap at 10pt with dense x-ticks.
- **Legend:** Never assume `loc="upper right"` is safe — always place the legend **outside** the axes with `bbox_to_anchor=(1.01, 1), loc="upper left"` and call `plt.tight_layout()` afterwards so the legend is included in the layout. Only use an in-axes placement when the corner is provably free of data and annotations.
- **Categorical x-axis tick labels:** Always rotate with `rotation=45, ha="right"` for labels longer than ~5 characters to prevent overlap with adjacent labels.
- **Confusion matrices:** Always call `ax.tick_params(axis="x", rotation=30, labelsize=10)` (and separately `ax.tick_params(axis="y", labelsize=10)`) so class names on the x-axis never overlap.
- **Multi-line axis labels:** Use at most one `\n` in an axis label; avoid `\n\n` (double newline) as it can push the label outside the figure boundary.

# Electronic Lab Notebook (ELN)

Every experiment has a corresponding entry in the [ELN](https://www.notion.so/202698419cbd8055bfc9db5bbf88b149?v=202698419cbd8092be3d000cc75d14e6&source=copy_link), which is a Notion page with the same name as the experiment folder, e.g., `EXP-26-IY023`. When an experiment is created, a corresponding ELN page should be created with the same name within the ELN database.

The ELN page for the experiment should contain:
  - **Introduction:** A brief description of the experiment and its objectives.
  - **Methods:** A detailed description of the methods used in the experiment, including any relevant code snippets or algorithms.
  - **Results & Discussion:** A summary of the results obtained from the experiment, including any relevant figures. Discussion of the results and their implications should also be included.
  - **Conclusion:** A brief conclusion summarising the key findings of the experiment and any future directions for research.
