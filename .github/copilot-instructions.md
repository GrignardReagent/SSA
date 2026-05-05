# Copilot Instructions for Stochastic Simulations

## Project Overview
This is a **stochastic simulation and machine learning** research project focused on gene expression modeling using the Stochastic Simulation Algorithm (SSA). The core workflow involves:
1. **Telegraph model simulation** (gene activation/deactivation with mRNA production)
2. **Time series classification/regression** using transformer models, LSTMs, and MLPs  
3. **Self-supervised learning** with pretraining on synthetic data and fine-tuning on experimental data

## Architecture & Data Flow

### Core Components
- `src/simulation/` - SSA implementations and telegraph model variants
- `src/models/` - ML models (TF_transformer.py, TF_pretraining.py) 
- `src/dataloaders/` - Data loading utilities (core.py handles train/val/test splits)
- `experiments/EXP-XX-IYXXX/` - Experiment runs with cluster job scripts (.sh) and Python scripts

### Key Data Flow
```
Telegraph Model Parameters → SSA Simulation → Time Series CSV → DataLoaders → ML Models → Classification/Regression
```

## Development Patterns

### Experiment Organization
- **Naming**: Experiments follow `EXP-{batch}-IY{number}` pattern (e.g., `EXP-26-IY017`)
- **Structure**: Each experiment contains:
  - `.py` script with hyperparameter sweeps and model training
  - `.sh` cluster job submission script (Grid Engine format)
  - Output files (.o, .e, .pe, .po for stdout/stderr)

## Critical Workflows

### Environment Setup
```bash
micromamba env create -f requirements.yml && micromamba activate stochastic_sim
pip install -e .  # Development install required for src/ modules
```

### Data Generation
Telegraph model simulations output CSV with format: `[label, timepoint_1, timepoint_2, ...]`

### Model Training
- Use `src/dataloaders/core.py:load_and_split_data()` for consistent train/val/test splits
- Transformer models expect 3D input: `(batch_size, seq_len, features)`
- LSTM models need reshape: `X.reshape((X.shape[0], X.shape[1], 1))`

### Cluster Jobs
Scripts use Grid Engine (`#$ -q gpu -l gpu=2`) for multi-GPU training. Job outputs log hyperparameter sweeps and best configurations.

## Project-Specific Conventions

### Import Style
- Local imports: `from simulation.simulate_telegraph_model import ...`
- Models: `from src.models.TF_transformer import TFTransformer, ModelCfg`

### Configuration Management
- Use dataclasses (`ModelCfg`) for model hyperparameters
- Experiment configs defined as lists of dictionaries for grid search

### File Organization
- **Notebooks** in `notebooks/` for exploration, **experiments** in `experiments/` for production runs
- **Data files** typically CSV format with descriptive names like `mRNA_trajectories_example.csv`

## Key Integration Points

### PyTorch Models
All models inherit from `nn.Module`. Use `torch.utils.data.DataLoader` with `TensorDataset` for training loops.

### Cluster Computing
Experiments designed for SLURM/Grid Engine clusters with GPU queues. Scripts handle multi-process simulation and distributed training.

When working on this codebase, always consider the full pipeline from simulation parameters → data generation → model training → evaluation.