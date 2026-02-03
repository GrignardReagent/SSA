import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent
# DATA_ROOT = script_dir / "data"
# RESULTS_PATH = DATA_ROOT / "IY011_simulation_parameters_sobol.csv"
# DATA_ROOT = script_dir / "temp_data_cv_variation"
# RESULTS_PATH = DATA_ROOT / "IY011_simulation_cv_parameters_sobol.csv"
# DATA_ROOT = script_dir / "temp_data_mu_variation"
# RESULTS_PATH = DATA_ROOT / "IY011_simulation_mu_parameters_sobol.csv"
DATA_ROOT = script_dir / "temp_data_t_ac_variation"
RESULTS_PATH = DATA_ROOT / "IY011_simulation_t_ac_parameters_sobol.csv"

df_params = pd.read_csv(RESULTS_PATH)

#plot mu_target vs mu_observed
plt.figure(figsize=(8, 6))
plt.scatter(df_params['mu_target'], df_params['mu_observed'], alpha=0.7, marker='.')
plt.plot([0, df_params['mu_target'].max()],
         [0, df_params['mu_target'].max()], 'r--', label='y=x')
plt.xlabel('Mu Target')
plt.ylabel('Mu Observed')
plt.title('Mu Target vs Mu Observed')
plt.legend()
plt.grid(True)
plt.savefig('IY011_mu_target_vs_mu_observed.png', dpi=300)
plt.show()

# plot cv_target vs cv_observed
plt.figure(figsize=(8, 6))
plt.scatter(df_params['cv_target'], df_params['cv_observed'], alpha=0.7, marker='.')
plt.plot([df_params['cv_target'].min(), df_params['cv_target'].max()],
         [df_params['cv_target'].min(), df_params['cv_target'].max()], 'r--', label='y=x')
plt.xlabel('CV Target')
plt.ylabel('CV Observed')
plt.title('CV Target vs CV Observed')
plt.legend()
plt.grid(True)
plt.savefig('IY011_cv_target_vs_cv_observed.png', dpi=300)
plt.show()

# plot t_ac_target vs t_ac_observed
plt.figure(figsize=(8, 6))
plt.scatter(df_params['t_ac_target'], df_params['t_ac_observed'], alpha=0.7, marker='.')
plt.plot([0, df_params['t_ac_target'].max()],
         [0, df_params['t_ac_target'].max()], 'r--', label='y=x')
plt.xlabel('T AC Target')
plt.ylabel('T AC Observed')
plt.title('T AC Target vs T AC Observed')
plt.legend()
plt.grid(True)
plt.savefig('IY011_t_ac_target_vs_t_ac_observed.png', dpi=300)
plt.show()