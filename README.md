# Stochastic Simulation Algorithm (SSA)

This repository contains all code for running the **Stochastic Simulation Algorithm (SSA)** to generate synthetic data for my PhD project. The SSA is used to simulate stochastic biochemical reactions and the algorithm built may be validated using the Fano factors, to ensure that the codes written are producing expected results, as you'll see in the notebooks.

## **🚀 Features**
- **Biological networks currently implemented:**
  - Simple **telegraph model**  
  - Telegraph model with **protein production and degradation**  

- **Future Plans:**
  - Transition to **Julia** for improved performance and scalability.

---

## 📥 Installation Guide
To install and set up the package, follow these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/GrignardReagent/SSA.git
cd SSA 
```

### 2️⃣ Set Up the Conda Environment
Ensure you have Mamba or Conda installed. Then create the environment:

```bash
micromamba env create -f requirements.yml
micromamba activate stochastic_sim
``` 

<!-- 
### **3️⃣ Install the Package**
This step ensures that Python recognises the package:

```bash
pip install -e .
``` -->

<!-- ## 📊 Usage
Once installed, you can run simulations and analyse data.

### Running a Simulation
Example for running the telegraph model:

```python
from ssa_simulation import telegraph_model_propensity

# Define parameters
propensities = [0.1, 0.05, 1.0, 0.5]  # Example values
population = [0, 1, 0]  # Initial state of the system

# Run the model
new_propensities = telegraph_model_propensity(propensities, population, 0.1, 0.05, 1.0, 0.5)
print(new_propensities)
```

### Statistical Analysis
Example for detecting steady state:

```python
from ssa_analysis import find_steady_state

time_points = np.linspace(0, 100, 1000)
mean_trajectory = np.random.randn(1000)  # Replace with real data

steady_state_time, steady_state_index = find_steady_state(time_points, mean_trajectory)
print(f"Steady state reached at time: {steady_state_time}")
``` -->

## 📜 Acknowledgements
This project is part of my PhD research, focusing on understanding and characterising metabolic oscillations of yeast cells using mathematical modelling and machine learning. 

For any issues or improvements, feel free to open a pull request or raise an issue! 🚀