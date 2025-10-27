import matplotlib.pyplot as plt
import pandas as pd
# read in the summary data
summary_df = pd.read_csv('summary_results.csv')
#plot mu_target vs mu_observed
plt.figure(figsize=(8, 6))
plt.scatter(summary_df['mu_target'], summary_df['mu_observed'], alpha=0.7)
plt.plot([0, summary_df['mu_target'].max()],
         [0, summary_df['mu_target'].max()], 'r--', label='y=x')
plt.xlabel('Mu Target')
plt.ylabel('Mu Observed')
plt.title('Mu Target vs Mu Observed')
plt.legend()
plt.grid(True)
plt.savefig('mu_target_vs_mu_observed.png', dpi=300)
plt.show()

# plot cv_target vs cv_observed
plt.figure(figsize=(8, 6))
plt.scatter(summary_df['cv_target'], summary_df['cv_observed'], alpha=0.7)
plt.plot([0, summary_df['cv_target'].max()],
         [0, summary_df['cv_target'].max()], 'r--', label='y=x')
plt.xlabel('CV Target')
plt.ylabel('CV Observed')
plt.title('CV Target vs CV Observed')
plt.legend()
plt.grid(True)
plt.savefig('cv_target_vs_cv_observed.png', dpi=300)
plt.show()

# plot t_ac_target vs t_ac_observed
plt.figure(figsize=(8, 6))
plt.scatter(summary_df['t_ac_target'], summary_df['t_ac_observed'], alpha=0.7)
plt.plot([0, summary_df['t_ac_target'].max()],
         [0, summary_df['t_ac_target'].max()], 'r--', label='y=x')
plt.xlabel('T AC Target')
plt.ylabel('T AC Observed')
plt.title('T AC Target vs T AC Observed')
plt.legend()
plt.grid(True)
plt.savefig('t_ac_target_vs_t_ac_observed.png', dpi=300)
plt.show()