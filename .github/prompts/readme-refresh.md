Scan the full repository and update the root `README.md` so it reflects the
current codebase.

Follow these repository-specific rules:

- Read `AGENTS.md` if present and follow its documentation conventions.
- Treat `src/` as the reusable library surface.
- Treat `experiments/EXP-YY-IYXXX/` folders as experiment records.
- Summarize each top-level experiment folder in one concise row if the README
  experiment index is stale or incomplete.
- Prefer current simulation APIs:
  - `simulation.mean_cv_t_ac.find_tilda_parameters`
  - `simulation.julia_simulate_telegraph_model.simulate_telegraph_model`
- Do not document deleted legacy APIs as current usage.
- Mention historical or archival code separately when relevant.
- Keep the README concise and practical for a new contributor.
- Keep the README ASCII-only.

Verification to run before finishing:

- `git diff --check README.md`
- Search `README.md` for stale deleted API names:
  `simulate_two_telegraph_model_systems`, `simulate_one_telegraph_model_system`,
  `run_simulation`, `extract_integer_timepoints`, `quick_find_parameters`,
  `find_sigma_sum`, `mean_var_autocorr`, `mean_cv_autocorr`,
  `gillespie_algorithm`
- Search `README.md` for non-ASCII characters.

Only edit `README.md` unless a very small workflow note is absolutely required.
Do not modify source code, notebooks, experiment outputs, generated data,
model checkpoints, or logs.
