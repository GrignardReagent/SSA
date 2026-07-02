# Fraisse VRAE Single-Cell Time-Series Data Probe

This folder documents a small, reproducible inspection of the data used in
Fraisse et al., *Representation learning of single-cell time-series with deep
variational autoencoders*. The goal is not to reproduce every model result in
the paper. Instead, the script here loads the public Zenodo dataset, follows the
authors' notebook conventions where relevant, and generates representative
plots for each raw time-series data source shown in the paper figures.

## Data and References

- Paper data record: `https://zenodo.org/records/20747946`
- Expected downloaded Zenodo archive: `raw/zenodo_20747946_data.zip`
- Zenodo metadata snapshot: `raw/zenodo_20747946_metadata.json`
- Downloaded archive MD5: `846b7e9b48a094e70f4a282a40f21a39`
- Reference notebook copied from the authors' GitHub repository:
  `reference/vrae_training.ipynb`
- Local paper copy used for figure inventory: `achille - vrae paper.pdf`

After downloading and extracting the Zenodo archive, the relevant public data
are under:

```text
raw/extracted/data for Representation learning of single cell time series with deep variational autoencoders/
```

The macOS metadata folders named `__MACOSX` are not data and were removed from
the extracted copy.

## Main Data Files Used

| Source | Local path | Notes |
| --- | --- | --- |
| Fraisse growth-antibiotic dataset | `raw/extracted/data for Representation learning of single cell time series with deep variational autoencoders/growth_antibiotic_dataset.csv` | Multivariate single-cell time series containing size, growth, division, fate, treatment, medium, and fluorescence measurements. |
| Tanouchi temperature dataset, 25C | `raw/extracted/data for Representation learning of single cell time series with deep variational autoencoders/temperature_dataset/Analysis_MC4100_25C/MC4100_25C` | External cell-length traces used by the paper for the temperature classification example. |
| Tanouchi temperature dataset, 27C | `raw/extracted/data for Representation learning of single cell time series with deep variational autoencoders/temperature_dataset/Analysis_MC4100_27C/MC4100_27C` | External cell-length traces used by the paper for the temperature classification example. |
| Tanouchi temperature dataset, 37C | `raw/extracted/data for Representation learning of single cell time series with deep variational autoencoders/temperature_dataset/Analysis_MC4100_37C/MC4100_37C` | External cell-length traces used by the paper for the temperature classification example. |

## Variables Used

The growth-antibiotic dataset is multivariate. The authors' base
`vrae_training.ipynb` workflow trains on one input channel:

- `FeretMax`: cell-size time series, used as the univariate VRAE input.

Other paper analyses and notebooks use additional channels:

- `MeanIntensity_gfp`: GFP fluorescence, used as an SOS-response readout.
- `MeanIntensity_mch`: mCherry / mKate fluorescence, used in downstream
  regression analyses.
- `GrowthRateFeretMax`: elongation-rate-related signal used in downstream
  regression analyses.
- `cellcycle_fate`: metadata used to truncate tracks at the first non-alive
  time point; it is not used as a VRAE input channel.

The base VRAE notebook keeps labels and metadata including `Medium`,
`Treatment`, `RepeatID`, and death timing. Its main embedding plots group cells
by medium: glycerol (`gly`), glucose (`glu`), or glucose + amino acids
(`gluaa`).

## Preprocessing In The Reference Notebook

The reference notebook first filters usable tracks, then splits complete cell
tracks into train and test sets with:

```python
train_test_split(..., test_size=0.1, random_state=42, shuffle=True)
```

It converts each variable-length trace into fixed 72-timepoint windows.
Training windows are overlapping, with stride 6 timepoints:

```python
for j in range(0, len(usable_data) - 72 + 1, 6):
    liste_x_train.append(usable_data[j:j+72])
```

Test windows are non-overlapping, with stride 72 timepoints:

```python
for j in range(0, len(usable_data) - 72 + 1, 72):
    liste_x_test.append(usable_data[j:j+72])
```

The notebook then applies `TimeSeriesScalerMeanVariance().fit_transform(...)`.
This is instance normalisation: each 72-point window is centred and scaled using
its own mean and standard deviation. This differs from
`sklearn.preprocessing.StandardScaler`, which estimates dataset-level
statistics during `fit()` and reuses them during `transform()`.

## Paper Figure Data Coverage

The table below maps the paper's time-series figures to the raw local data
source and the representative figure generated in this folder.

| Paper figure | Time-series source | Local source path | Generated figure |
| --- | --- | --- | --- |
| Figure 1 | Fraisse control-cell size trajectories from `FeretMax` across glycerol, glucose, and glucose + amino acids | `growth_antibiotic_dataset.csv`, filtered to `gly_control_*`, `glu_control_*`, and `gluaa_control_*` | `figures/IY027_fraisse_vrae_control_time_series.png` and `figures/IY027_fraisse_vrae_72_point_windows.png` |
| Figure 2 | Fraisse glucose cell-size trajectories from `FeretMax` during control, ciprofloxacin, tetracycline, and ciprofloxacin + tetracycline exposure | `growth_antibiotic_dataset.csv`, filtered to `glu_control_*`, `glu_cip_*`, `glu_tet_*`, and `glu_ciptet_*` | `figures/IY027_fraisse_vrae_glucose_antibiotic_size_time_series.png` |
| Figure 3 | Fraisse SOS/GFP fluorescence from `MeanIntensity_gfp` and paired size + SOS trajectories from `FeretMax` + `MeanIntensity_gfp` | `growth_antibiotic_dataset.csv`, filtered to the same glucose antibiotic datasets as Figure 2 | `figures/IY027_fraisse_vrae_sos_gfp_fluorescence_time_series.png` and `figures/IY027_fraisse_vrae_size_sos_multivariate_examples.png` |
| Figure 4 | External Tanouchi temperature-dataset cell-length trajectories at 25C, 27C, and 37C | `temperature_dataset/Analysis_MC4100_25C/MC4100_25C`, `temperature_dataset/Analysis_MC4100_27C/MC4100_27C`, and `temperature_dataset/Analysis_MC4100_37C/MC4100_37C` | `figures/IY027_tanouchi_temperature_cell_length_time_series.png` |
| Supplementary Figure S5 | Fraisse regression-associated constitutive mKate fluorescence and elongation-rate signals from `MeanIntensity_mch` and `GrowthRateFeretMax` | `growth_antibiotic_dataset.csv`, filtered to the control datasets | `figures/IY027_fraisse_vrae_mkate_elongation_time_series.png` |

These generated figures are representative data-source probes. They are not
intended to reproduce the paper's VRAE reconstructions, embeddings, ROC curves,
or feature-importance panels.

## Generated Figures

- `figures/IY027_fraisse_vrae_control_time_series.png`
- `figures/IY027_fraisse_vrae_72_point_windows.png`
- `figures/IY027_fraisse_vrae_glucose_antibiotic_size_time_series.png`
- `figures/IY027_fraisse_vrae_sos_gfp_fluorescence_time_series.png`
- `figures/IY027_fraisse_vrae_size_sos_multivariate_examples.png`
- `figures/IY027_fraisse_vrae_mkate_elongation_time_series.png`
- `figures/IY027_tanouchi_temperature_cell_length_time_series.png`

## Control Tracks Loaded By The Probe

The base VRAE-style control analysis uses the same seven control datasets as
the reference notebook. A track is counted here if it remains finite after the
notebook-style filtering and is longer than one 72-point window.

| Dataset | Usable tracks | Median length / timepoints | Selected example length / timepoints |
| --- | --- | --- | --- |
| `gly_control_1` | 184 | 285.0 | 285 |
| `gly_control_2` | 368 | 227.0 | 227 |
| `gly_control_3` | 291 | 186.0 | 186 |
| `glu_control_1` | 76 | 229.0 | 229 |
| `glu_control_2` | 333 | 288.0 | 288 |
| `gluaa_control_1` | 274 | 265.0 | 265 |
| `gluaa_control_2` | 325 | 288.0 | 288 |

Total usable control tracks: `1851`.

## Reproduce

Run from the repository root:

```bash
micromamba run -n stochastic_sim python experiments/EXP-26-IY027/fraisse_vrae_data_probe/IY027_plot_fraisse_vrae_time_series.py
```

The script writes all PNG files into:

```text
experiments/EXP-26-IY027/fraisse_vrae_data_probe/figures/
```
