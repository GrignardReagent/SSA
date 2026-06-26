"""Paired experimental mCherry/GFP time-series loading utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalise_tf_name(group_name: str) -> str:
    """Remove channel prefixes and normalise TF names for grouping."""
    return re.sub(r"^ch\d+_", "", str(group_name)).upper()


def discover_paired_files(data_dir: str | Path) -> pd.DataFrame:
    """Find file stems with both mCherry and GFP time-series CSVs."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    gfp = {
        p.stem.replace("_GFP_time_series", ""): p
        for p in data_dir.glob("*_GFP_time_series.csv")
    }
    mcherry = {
        p.stem.replace("_mCherry_time_series", ""): p
        for p in data_dir.glob("*_mCherry_time_series.csv")
    }

    rows = []
    for key in sorted(set(gfp) & set(mcherry)):
        experiment = key.split("_group_")[0]
        group = key.split("_group_", 1)[1]
        rows.append(
            {
                "key": key,
                "experiment": experiment,
                "group": group,
                "tf": normalise_tf_name(group),
                "mcherry_path": mcherry[key],
                "gfp_path": gfp[key],
                "n_rows": sum(1 for _ in open(mcherry[key])) - 1,
            }
        )
    return pd.DataFrame(rows)


def load_selected_pairs(
    pair_table: pd.DataFrame,
    selected_classes: list[str],
    meta_cols: list[str] | tuple[str, ...] = ("id", "group", "experiment"),
):
    """Load paired mCherry/GFP traces for selected TF classes."""
    m_arrays, g_arrays, metadata_rows = [], [], []
    selected_table = pair_table[pair_table["tf"].isin(selected_classes)].copy()

    for _, row in selected_table.iterrows():
        df_m = pd.read_csv(row["mcherry_path"])
        df_g = pd.read_csv(row["gfp_path"])
        time_cols_m = [c for c in df_m.columns if c not in meta_cols]
        time_cols_g = [c for c in df_g.columns if c not in meta_cols]

        # Keep only cells observed in both channels so paired rows stay aligned.
        common_ids = sorted(set(df_m["id"]) & set(df_g["id"]))
        df_m = df_m[df_m["id"].isin(common_ids)].set_index("id").loc[common_ids]
        df_g = df_g[df_g["id"].isin(common_ids)].set_index("id").loc[common_ids]
        m_arrays.append(df_m[time_cols_m].to_numpy(float))
        g_arrays.append(df_g[time_cols_g].to_numpy(float))

        for cell_id in common_ids:
            metadata_rows.append(
                {
                    "cell_id": cell_id,
                    "class_name": row["tf"],
                    "tf": row["tf"],
                    "group": row["group"],
                    "experiment": row["experiment"],
                    "source_key": row["key"],
                }
            )
        print(
            f"{row['tf']:8s} {row['experiment']:>5s} "
            f"{row['group']:<18s}: {len(common_ids):4d} paired cells"
        )

    min_tp_m = min(arr.shape[1] for arr in m_arrays)
    min_tp_g = min(arr.shape[1] for arr in g_arrays)
    X_m = np.vstack([arr[:, :min_tp_m] for arr in m_arrays])
    X_g = np.vstack([arr[:, :min_tp_g] for arr in g_arrays])
    metadata = pd.DataFrame(metadata_rows)
    class_to_label = {name: idx for idx, name in enumerate(selected_classes)}
    metadata["label"] = metadata["class_name"].map(class_to_label).astype(int)
    return X_m, X_g, metadata, min_tp_m, min_tp_g, class_to_label
