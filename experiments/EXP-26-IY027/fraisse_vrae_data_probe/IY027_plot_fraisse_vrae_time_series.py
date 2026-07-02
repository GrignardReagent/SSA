"""Plot representative time series from the Fraisse et al. VRAE dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXP_DIR = Path(__file__).resolve().parent
RAW_DIR = EXP_DIR / "raw" / "extracted"
FIG_DIR = EXP_DIR / "figures"
DATASET_ROOT = RAW_DIR / "data for Representation learning of single cell time series with deep variational autoencoders"
GROWTH_DATA_PATH = DATASET_ROOT / "growth_antibiotic_dataset.csv"
CONTROL_DATASETS = (
    "gly_control_1",
    "gly_control_2",
    "gly_control_3",
    "glu_control_1",
    "glu_control_2",
    "gluaa_control_1",
    "gluaa_control_2",
)
GLUCOSE_ANTIBIOTIC_DATASETS = (
    "glu_control_1",
    "glu_control_2",
    "glu_cip_1",
    "glu_cip_2",
    "glu_tet_1",
    "glu_tet_2",
    "glu_ciptet_1",
    "glu_ciptet_2",
)
GROUP_COLS = (
    "Position",
    "ParentTrackHeadIndices",
    "Medium",
    "Treatment",
    "RepeatID",
    "RepeatDate",
    "fate",
    "DeathSubtype",
)
TIME_STEP_H = 5.0 / 60.0
ANTIBIOTIC_START_H = 2.0
ANTIBIOTIC_END_H = 14.0
MAX_UPSTREAM_LENGTH = 288
WINDOW_LENGTH = 72


@dataclass(frozen=True)
class Track:
    """Single-cell FeretMax time series prepared with the upstream VRAE rules."""

    medium: str
    treatment: str
    replicate: str
    track_id: tuple
    death_timepoint: int
    values: np.ndarray

    @property
    def dataset_name(self) -> str:
        return f"{self.medium}_{self.treatment}_{self.replicate}"


@dataclass(frozen=True)
class MultiSignalTrack:
    """Single-cell track carrying one or more paper-relevant time-series signals."""

    medium: str
    treatment: str
    replicate: str
    track_id: tuple
    signals: dict[str, np.ndarray]

    @property
    def dataset_name(self) -> str:
        return f"{self.medium}_{self.treatment}_{self.replicate}"


def fill_internal_nans(values: np.ndarray) -> np.ndarray:
    """Fill internal missing FeretMax values using the upstream local rule."""

    values = values.astype(float, copy=True)

    # Match the notebook's local interpolation rule for missing internal points.
    for idx in range(1, len(values) - 1):
        if np.isnan(values[idx]):
            values[idx] = np.nanmean([values[idx - 1], values[idx + 1]])

    return values


def first_nan_index(values: np.ndarray) -> int:
    """Return the first NaN index or the full series length if no NaN exists."""

    nan_indices = np.flatnonzero(np.isnan(values))
    if len(nan_indices) == 0:
        return len(values)

    return int(nan_indices[0])


def first_non_alive_timepoint(fates: pd.Series) -> int:
    """Return the first non-alive frame, or the full track length if none exists."""

    fate_values = fates.ffill().bfill().astype(str).to_numpy()
    non_alive = np.flatnonzero(fate_values != "alive")

    if len(non_alive) == 0:
        return len(fate_values)

    return int(non_alive[0])


def read_growth_antibiotic_data() -> pd.DataFrame:
    """Load the growth-antibiotic dataframe distributed with the paper."""

    return pd.read_pickle(GROWTH_DATA_PATH)


def load_upstream_control_tracks() -> list[Track]:
    """Load control tracks selected by the VRAE notebook."""

    df = read_growth_antibiotic_data()
    tracks: list[Track] = []

    for dataset_name in CONTROL_DATASETS:
        medium, treatment, replicate = dataset_name.split("_")

        # Use the same seven control experiments and variables as vrae_training.ipynb.
        exp_df = df[
            (df["Medium"] == medium)
            & (df["Treatment"] == treatment)
            & (df["RepeatID"].astype(str) == replicate)
        ].copy()
        exp_df = exp_df.sort_values("Time")

        for track_id, track_df in exp_df.groupby(list(GROUP_COLS), dropna=False, sort=False):
            values = fill_internal_nans(pd.to_numeric(track_df["FeretMax"], errors="coerce").to_numpy())
            death_timepoint = first_non_alive_timepoint(track_df["cellcycle_fate"])
            usable = values[: min(death_timepoint, MAX_UPSTREAM_LENGTH)]

            # The upstream notebook only keeps tracks longer than one 72-point window.
            if len(usable) > WINDOW_LENGTH and np.isfinite(usable).all():
                tracks.append(
                    Track(
                        medium=medium,
                        treatment=treatment,
                        replicate=replicate,
                        track_id=track_id,
                        death_timepoint=death_timepoint,
                        values=usable,
                    )
                )

    return tracks


def load_growth_signal_tracks(
    dataset_names: tuple[str, ...],
    signal_columns: tuple[str, ...],
    max_timepoints: int,
    min_length: int,
) -> list[MultiSignalTrack]:
    """Load paper-relevant growth-antibiotic tracks for selected signals."""

    df = read_growth_antibiotic_data()
    tracks: list[MultiSignalTrack] = []

    for dataset_name in dataset_names:
        medium, treatment, replicate = dataset_name.split("_")
        exp_df = df[
            (df["Medium"] == medium)
            & (df["Treatment"] == treatment)
            & (df["RepeatID"].astype(str) == replicate)
        ].copy()
        exp_df = exp_df.sort_values("Time")

        for track_id, track_df in exp_df.groupby(list(GROUP_COLS), dropna=False, sort=False):
            raw_signals = {
                column: pd.to_numeric(track_df[column], errors="coerce").to_numpy(dtype=float)
                for column in signal_columns
            }
            stop = min(max_timepoints, *(first_nan_index(values) for values in raw_signals.values()))

            signals = {
                column: fill_internal_nans(values[:stop])
                for column, values in raw_signals.items()
            }

            if all(len(values) >= min_length and np.isfinite(values).all() for values in signals.values()):
                tracks.append(
                    MultiSignalTrack(
                        medium=medium,
                        treatment=treatment,
                        replicate=replicate,
                        track_id=track_id,
                        signals=signals,
                    )
                )

    return tracks


def standardise_window(values: np.ndarray) -> np.ndarray:
    """Mean-centre and variance-scale one 72-point VRAE input window."""

    window = values[:WINDOW_LENGTH].astype(float, copy=True)
    std = window.std()

    if std == 0:
        return window - window.mean()

    return (window - window.mean()) / std


def select_representative_tracks(tracks: list[Track]) -> list[Track]:
    """Select one deterministic representative track per upstream control dataset."""

    selected: list[Track] = []

    for dataset_name in CONTROL_DATASETS:
        candidates = [track for track in tracks if track.dataset_name == dataset_name]
        if not candidates:
            continue

        # Pick a long track so the raw trace visibly shows the pre-truncation dynamics.
        selected.append(max(candidates, key=lambda track: len(track.values)))

    return selected


def select_one_track_per_treatment(
    tracks: list[MultiSignalTrack],
    treatments: tuple[str, ...],
    ranking_signal: str,
) -> list[MultiSignalTrack]:
    """Select one deterministic representative track for each treatment."""

    selected: list[MultiSignalTrack] = []

    for treatment in treatments:
        candidates = [track for track in tracks if track.treatment == treatment]
        if not candidates:
            continue
        selected.append(max(candidates, key=lambda track: len(track.signals[ranking_signal])))

    return selected


def log_transform(values: np.ndarray) -> np.ndarray:
    """Log-transform positive fluorescence values while guarding against zeros."""

    return np.log(np.clip(values.astype(float), a_min=1e-9, a_max=None))


def plot_raw_tracks(selected_tracks: list[Track]) -> Path:
    """Plot one raw upstream-style usable track per control dataset."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    palette = dict(zip(CONTROL_DATASETS, sns.color_palette("colorblind", len(CONTROL_DATASETS))))
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    for track in selected_tracks:
        time_h = np.arange(len(track.values)) * TIME_STEP_H
        ax.plot(
            time_h,
            track.values,
            label=track.dataset_name,
            color=palette[track.dataset_name],
            linewidth=1.6,
        )

    ax.set_title("Representative Fraisse VRAE control tracks", fontsize=14)
    ax.set_xlabel("Time / h", fontsize=12)
    ax.set_ylabel("Feret max length / um", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10, title="Dataset")

    out_path = FIG_DIR / "IY027_fraisse_vrae_control_time_series.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_glucose_antibiotic_size_tracks(selected_tracks: list[MultiSignalTrack]) -> Path:
    """Plot glucose-condition cell-size traces used for antibiotic classification."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    palette = dict(zip(("control", "cip", "tet", "ciptet"), sns.color_palette("colorblind", 4)))
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    ax.axvspan(
        ANTIBIOTIC_START_H,
        ANTIBIOTIC_END_H,
        color="0.85",
        alpha=0.65,
        label="Antibiotic exposure window",
        zorder=0,
    )
    for track in selected_tracks:
        values = track.signals["FeretMax"]
        time_h = np.arange(len(values)) * TIME_STEP_H
        ax.plot(
            time_h,
            values,
            label=track.treatment,
            color=palette[track.treatment],
            linewidth=1.5,
        )

    ax.set_title("Glucose antibiotic-exposure cell-size trajectories", fontsize=14)
    ax.set_xlabel("Time / h", fontsize=12)
    ax.set_ylabel("Feret max length / um", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

    out_path = FIG_DIR / "IY027_fraisse_vrae_glucose_antibiotic_size_time_series.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sos_gfp_tracks(selected_tracks: list[MultiSignalTrack]) -> Path:
    """Plot log GFP/SOS fluorescence traces from glucose antibiotic conditions."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    palette = dict(zip(("control", "cip", "tet", "ciptet"), sns.color_palette("colorblind", 4)))
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    ax.axvspan(
        ANTIBIOTIC_START_H,
        ANTIBIOTIC_END_H,
        color="0.85",
        alpha=0.65,
        label="Antibiotic exposure window",
        zorder=0,
    )
    for track in selected_tracks:
        values = log_transform(track.signals["MeanIntensity_gfp"])
        time_h = np.arange(len(values)) * TIME_STEP_H
        ax.plot(
            time_h,
            values,
            label=track.treatment,
            color=palette[track.treatment],
            linewidth=1.5,
        )

    ax.set_title("Glucose SOS/GFP fluorescence trajectories", fontsize=14)
    ax.set_xlabel("Time / h", fontsize=12)
    ax.set_ylabel("log GFP fluorescence / a.u.", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

    out_path = FIG_DIR / "IY027_fraisse_vrae_sos_gfp_fluorescence_time_series.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_size_sos_multivariate_tracks(selected_tracks: list[MultiSignalTrack]) -> Path:
    """Plot paired size and SOS/GFP traces used by the multivariate workflow."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    palette = dict(zip(("control", "cip", "tet", "ciptet"), sns.color_palette("colorblind", 4)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_size, ax_sos = axes

    for ax in axes:
        ax.axvspan(ANTIBIOTIC_START_H, ANTIBIOTIC_END_H, color="0.85", alpha=0.65, zorder=0)
        ax.tick_params(labelsize=10)

    for track in selected_tracks:
        time_h = np.arange(len(track.signals["FeretMax"])) * TIME_STEP_H
        ax_size.plot(
            time_h,
            track.signals["FeretMax"],
            label=track.treatment,
            color=palette[track.treatment],
            linewidth=1.4,
        )
        ax_sos.plot(
            time_h,
            log_transform(track.signals["MeanIntensity_gfp"]),
            label=track.treatment,
            color=palette[track.treatment],
            linewidth=1.4,
        )

    ax_size.set_title("Paired size and SOS/GFP trajectories", fontsize=14)
    ax_size.set_ylabel("Feret max length / um", fontsize=12)
    ax_sos.set_xlabel("Time / h", fontsize=12)
    ax_sos.set_ylabel("log GFP fluorescence / a.u.", fontsize=12)
    ax_size.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

    out_path = FIG_DIR / "IY027_fraisse_vrae_size_sos_multivariate_examples.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mkate_elongation_tracks(selected_tracks: list[MultiSignalTrack]) -> Path:
    """Plot supplementary regression signals: mKate fluorescence and elongation rate."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    labels = [track.dataset_name for track in selected_tracks]
    palette = dict(zip(labels, sns.color_palette("colorblind", len(labels))))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    ax_mkate, ax_elongation = axes

    for track in selected_tracks:
        label = track.dataset_name
        time_h = np.arange(len(track.signals["MeanIntensity_mch"])) * TIME_STEP_H
        ax_mkate.plot(
            time_h,
            log_transform(track.signals["MeanIntensity_mch"]),
            label=label,
            color=palette[label],
            linewidth=1.3,
        )
        ax_elongation.plot(
            time_h,
            track.signals["GrowthRateFeretMax"],
            label=label,
            color=palette[label],
            linewidth=1.3,
        )

    ax_mkate.set_title("Supplementary mKate fluorescence and elongation-rate traces", fontsize=14)
    ax_mkate.set_ylabel("log mKate fluorescence / a.u.", fontsize=12)
    ax_elongation.set_xlabel("Time / h", fontsize=12)
    ax_elongation.set_ylabel("Elongation rate / 1/h", fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
    ax_mkate.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

    out_path = FIG_DIR / "IY027_fraisse_vrae_mkate_elongation_time_series.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_standardised_windows(selected_tracks: list[Track]) -> Path:
    """Plot the first standardized 72-point VRAE input window per dataset."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    palette = dict(zip(CONTROL_DATASETS, sns.color_palette("colorblind", len(CONTROL_DATASETS))))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for track in selected_tracks:
        time_h = np.arange(WINDOW_LENGTH) * TIME_STEP_H
        ax.plot(
            time_h,
            standardise_window(track.values),
            label=track.dataset_name,
            color=palette[track.dataset_name],
            linewidth=1.6,
        )

    ax.set_title("First 72-point VRAE input windows", fontsize=14)
    ax.set_xlabel("Time / h", fontsize=12)
    ax.set_ylabel("Standardised Feret max length / a.u.", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10, title="Dataset")

    out_path = FIG_DIR / "IY027_fraisse_vrae_72_point_windows.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_temperature_sequences() -> dict[int, list[np.ndarray]]:
    """Load Tanouchi temperature cell-length traces using the upstream convention."""

    temperature_root = DATASET_ROOT / "temperature_dataset"
    sequences: dict[int, list[np.ndarray]] = {25: [], 27: [], 37: []}

    for temperature in sequences:
        folder = temperature_root / f"Analysis_MC4100_{temperature}C" / f"MC4100_{temperature}C"
        for file_path in sorted(folder.glob("*.txt")):
            if file_path.name.startswith("._"):
                continue
            df = pd.read_csv(file_path, delimiter=",", header=None)
            cell_length = pd.to_numeric(df[2], errors="coerce").to_numpy(dtype=float)[::5]
            if len(cell_length) >= WINDOW_LENGTH and np.isfinite(cell_length[:WINDOW_LENGTH]).all():
                sequences[temperature].append(cell_length)

    return sequences


def plot_temperature_cell_length_tracks(sequences: dict[int, list[np.ndarray]]) -> Path:
    """Plot representative normalized 6-hour temperature-dataset cell-length traces."""

    sns.set_theme(style="whitegrid", palette="colorblind", font="sans-serif")
    palette = dict(zip((25, 27, 37), sns.color_palette("colorblind", 3)))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for temperature, traces in sequences.items():
        trace = max(traces, key=len)[:WINDOW_LENGTH]
        time_h = np.arange(WINDOW_LENGTH) * TIME_STEP_H
        ax.plot(
            time_h,
            standardise_window(trace),
            label=f"{temperature}C",
            color=palette[temperature],
            linewidth=1.6,
        )

    ax.set_title("Tanouchi temperature-dataset cell-length windows", fontsize=14)
    ax.set_xlabel("Time / h", fontsize=12)
    ax.set_ylabel("Standardised cell length / a.u.", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10, title="Temperature")

    out_path = FIG_DIR / "IY027_tanouchi_temperature_cell_length_time_series.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Generate representative VRAE data-probe plots."""

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_upstream_control_tracks()
    selected_tracks = select_representative_tracks(tracks)

    if len(selected_tracks) != len(CONTROL_DATASETS):
        missing = sorted(set(CONTROL_DATASETS) - {track.dataset_name for track in selected_tracks})
        raise RuntimeError(f"Missing representative tracks for: {missing}")

    antibiotic_tracks = load_growth_signal_tracks(
        GLUCOSE_ANTIBIOTIC_DATASETS,
        ("FeretMax", "MeanIntensity_gfp"),
        max_timepoints=240,
        min_length=168,
    )
    selected_antibiotic_tracks = select_one_track_per_treatment(
        antibiotic_tracks,
        ("control", "cip", "tet", "ciptet"),
        ranking_signal="FeretMax",
    )
    if len(selected_antibiotic_tracks) != 4:
        missing = sorted({"control", "cip", "tet", "ciptet"} - {track.treatment for track in selected_antibiotic_tracks})
        raise RuntimeError(f"Missing glucose antibiotic tracks for: {missing}")

    regression_tracks = load_growth_signal_tracks(
        CONTROL_DATASETS,
        ("MeanIntensity_mch", "GrowthRateFeretMax"),
        max_timepoints=168,
        min_length=WINDOW_LENGTH,
    )
    selected_regression_tracks = select_representative_tracks(
        [
            Track(
                medium=track.medium,
                treatment=track.treatment,
                replicate=track.replicate,
                track_id=track.track_id,
                death_timepoint=len(track.signals["MeanIntensity_mch"]),
                values=track.signals["MeanIntensity_mch"],
            )
            for track in regression_tracks
        ]
    )
    selected_regression_ids = {track.track_id for track in selected_regression_tracks}
    selected_regression_signal_tracks = [
        track for track in regression_tracks if track.track_id in selected_regression_ids
    ]

    temperature_sequences = load_temperature_sequences()

    figure_paths = [
        plot_raw_tracks(selected_tracks),
        plot_standardised_windows(selected_tracks),
        plot_glucose_antibiotic_size_tracks(selected_antibiotic_tracks),
        plot_sos_gfp_tracks(selected_antibiotic_tracks),
        plot_size_sos_multivariate_tracks(selected_antibiotic_tracks),
        plot_mkate_elongation_tracks(selected_regression_signal_tracks),
        plot_temperature_cell_length_tracks(temperature_sequences),
    ]

    print(f"Loaded {len(tracks)} upstream-style usable tracks.")
    print(f"Loaded {len(antibiotic_tracks)} glucose antibiotic size/SOS tracks.")
    print(f"Loaded {sum(len(v) for v in temperature_sequences.values())} temperature-dataset traces.")
    for figure_path in figure_paths:
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
