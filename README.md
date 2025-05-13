# Stochastic Simulation Algorithm (SSA) & Time Series Analysis

This repository contains all code for running the **Stochastic Simulation Algorithm (SSA)** to generate synthetic time series data for analyzing gene expression models, with a focus on machine learning applications for classification and regression tasks. The SSA is used to simulate stochastic biochemical reactions, and the implementation is validated using Fano factor analysis.

## **🚀 Features**
- **Biological networks implemented:**
  - **Telegraph model** - A simple gene activation/deactivation model with mRNA production
  - **Telegraph model with protein dynamics** - Including protein production and degradation
- **Machine learning models for time series classification:**
  - LSTM-based neural networks
  - Transformer models
  - Multilayer Perceptrons (MLP)
- **Advanced analysis tools:**
  - Auto-correlation analysis
  - Feature importance analysis
  - Steady state detection

---

## 📥 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/GrignardReagent/SSA.git
cd SSA 
```

### 2️⃣ Set Up the Environment
Using micromamba or conda:

```bash
# If using micromamba
micromamba env create -f requirements.yml
micromamba activate stochastic_sim

# If using conda
conda env create -f requirements.yml
conda activate stochastic_sim
```

### 3️⃣ Install the Package in Development Mode
This ensures Python can find all modules in the package:

```bash
pip install -e .
```

## 📊 Simulating Synthetic Data

The repository provides tools for generating synthetic time series data using the Stochastic Simulation Algorithm (SSA).

### Basic Telegraph Model Simulation

The package provides two main functions for running simulations:

1. `simulate_two_telegraph_model_systems`: Simulates two different systems (e.g., normal and stressed conditions)
2. `simulate_one_telegraph_model_system`: Simulates a single system with given parameters

#### Example: Simulating Two Systems (Normal and Stressed Conditions)

The simplest way to generate data for two different conditions is through the `mRNA_traj_sim.ipynb` notebook:

```python
from simulation.simulate_telegraph_model import simulate_two_telegraph_model_systems
import numpy as np

# Define parameters for normal and stressed conditions
parameter_sets = [
    {"sigma_u": 18.0, "sigma_b": 0.01, "rho": 5, "d": 1.0, "label": 0},  # Stressed condition
    {"sigma_u": 9.0, "sigma_b": 0.02, "rho": 10, "d": 1.0, "label": 1},  # Normal condition
]

# Time points for simulation (144 time points for 12 hours with 5-min intervals)
time_points = np.arange(0, 144.0, 1.0)

# Number of trajectories to generate per condition
size = 200

# Run simulation
df_results = simulate_two_telegraph_model_systems(parameter_sets, time_points, size)

# Save dataset for ML classification
df_results.to_csv("data/mRNA_trajectories_example.csv", index=False)
```

#### Example: Simulating a Single System

You can also simulate a single system with specific parameters:

```python
from simulation.simulate_telegraph_model import simulate_one_telegraph_model_system
import numpy as np

# Define parameters for a single condition
parameter_set = {"sigma_u": 0.1, "sigma_b": 0.1, "rho": 10.0, "d": 1.0, "label": 0}

# Time points for simulation
time_points = np.arange(0, 144.0, 1.0)

# Number of trajectories to generate
size = 200

# Run simulation for a single system
df_results = simulate_one_telegraph_model_system(parameter_set, time_points, size)

# Save dataset
df_results.to_csv("data/mRNA_trajectories_single_system.csv", index=False)
```

Both simulators utilize multiprocessing to speed up simulations across multiple CPU cores.

## 📓 Notebook Guide

The repository contains various notebooks for different aspects of stochastic simulation and time series analysis:

### Core Notebooks (Ready to Use)
- **mRNA_traj_sim.ipynb** - Generate synthetic time series data using the telegraph model
- **fano_factor_plots.ipynb** - Validate SSA implementation through Fano factor analysis
- **LSTM_classification.ipynb** - Time series classification using LSTM models
- **MLP.ipynb** - Simple neural network approach to time series classification
- **transformer.ipynb** - Use transformer models for time series classification
- **fix_mean_and_variance.ipynb** - Generate time series with controlled statistical properties

### Work In Progress (WIP) Notebooks
The repository contains several work-in-progress notebooks focusing on:

- Exploratory Data Analysis (EDA) with LSTM models 
- Regression tasks using time series data
- Classifier hyperparameter tuning and optimization
- Auto-correlation and cross-correlation analysis
- Classification benchmarking across different models
- Data distribution and statistical pattern analysis

### Planned Notebooks (TODO)
Future notebooks are planned to explore:

- Techniques for adding different noise patterns to simulations
- Feature importance analysis and interpretation

### Experimental Framework
Long-running simulations and systematic model finetuning experiments are organized in the `experiments/` folder. These experiments follow a structured approach to parameter exploration and are designed to be reproducible. The naming conventions for these experiments are still evolving and will be standardized in the future.

All experimental details and results are/will be systematically documented in an Electronic Lab Notebook (ELN) that will be made available alongside this repository to ensure full reproducibility.

## 📂 Source Code Organization

The `src` directory is organized into the following subdirectories:

### Subdirectories
- **simulation/** - Core simulation capabilities and models:
  - `simulate_telegraph_model.py` - Implementation of the Stochastic Simulation Algorithm (SSA) for telegraph model
- **classifiers/** - Classification algorithms for time series data:
  - `svm_classifier.py` - Support Vector Machine classifier
  - `random_forest_classifier.py` - Random Forest classifier 
  - `logistic_regression_classifier.py` - Logistic Regression classifier
  - `random_classifier.py` - Random baseline classifier
- **stats/** - Statistical analysis tools:
  - `report.py` - Statistical reporting functions
  - `autocorrelation.py` - Auto-correlation and cross-correlation analysis
- **models/** - Neural network model implementations:
  - `MLP.py` - Multilayer Perceptron implementation
  - `LSTM.py` - LSTM classifier and regressor implementations
  - `transformer.py` - Transformer-based time series models
- **utils/** - Utility functions for data loading and preprocessing:
  - `load_data.py` - Data loading utilities
  - `set_seed.py` - Random seed setting for reproducibility
  - `steady_state.py` - Steady state detection functions
- **visualisation/** - Plotting and visualization tools:
  - `plots.py` - Functions for plotting trajectories, distributions, and analysis results

## 📜 Acknowledgements
This project is part of my PhD research, focusing on understanding and characterizing metabolic oscillations of yeast cells using mathematical modeling and machine learning. 

For any issues or improvements, feel free to open a pull request or raise an issue! 🚀