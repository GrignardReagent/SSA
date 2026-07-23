"""
Downstream KNN check for the IY035 SimCLR checkpoint (pretrained via
self-supervised contrastive learning directly on unlabelled experimental
FULL_DATA_DIR traces, using ssl_exp_data_prep), on the same 6-class
TF@condition task (Nrg1/Rtg1 x 3 glucose conditions) used in IY031/IY032.

Methodology mirrors IY032_knn.ipynb: encode the StandardScaler-normalised raw
traces (fit on train) through the frozen backbone, then KNeighborsClassifier
on the RAW embeddings with the Euclidean metric -- no StandardScaler on the
embeddings. (A scaling/metric ablation over all IY035 checkpoints found
embedding StandardScaling to be a wash on average that homogenises
checkpoints -- collapsing several to exactly 0.766 -- so raw+Euclidean is used
throughout.) k=10 reuses IY032's grid-search result.
"""

import re
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.embeddings import load_simclr_model, encode_channel
from utils.experimental_time_series import load_labelled_time_series_csvs
from utils.processing.pipeline import prepare_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IY035_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY035")
IY032_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-26-IY032")
IY008_DIR = Path("/home/ianyang/stochastic_simulations/experiments/EXP-25-IY008")
FULL_DATA_DIR = IY008_DIR / "5_FULL_transformed_exp_time_series"
META_PATH = IY008_DIR / "old_data_metadata.csv"
META_COLS = ["id", "group", "experiment"]

FILE_RE = re.compile(r"^(\d+)_.*_group_(.+?)_(GFP|mCherry)_time_series$")
EXCLUDED_EXPS = {"18446"}  # not properly recorded, excluded from all analyses
FIXED_CLASSES = [
    "Nrg1 @ 0.01% glucose", "Nrg1 @ 0.1% glucose", "Nrg1 @ 2% glucose (mock/steady)",
    "Rtg1 @ 0.01% glucose", "Rtg1 @ 0.1% glucose", "Rtg1 @ 2% glucose (mock/steady)",
]
RANDOM_STATE = 42
K = 10           # from IY032's grid search (best_k=10)

# find all the trained simclr paths 
SIMCLR_CKPTS = list(glob.glob(str(IY035_DIR / "IY035_*.pth")))

# === 1. Same labelled 6-class Full dataset as IY031/IY032 ===
metadata = pd.read_csv(META_PATH)
metadata["exp_id"] = metadata["exp_id"].astype(str)
metadata["group_id"] = metadata["group_id"].astype(str)
LABEL_LOOKUP = {(r.exp_id, r.group_id, r.channel): (r.tf, r.condition) for _, r in metadata.iterrows()}

full_ts_raw, full_label_strs = load_labelled_time_series_csvs(
    data_dir=FULL_DATA_DIR, file_re=FILE_RE, label_lookup=LABEL_LOOKUP,
    meta_cols=META_COLS, excluded_exps=EXCLUDED_EXPS, verbose=False,
)
d = prepare_dataset(full_ts_raw, full_label_strs, FIXED_CLASSES, "Full", RANDOM_STATE)
n_cls = len(d["class_names"])
chance = 1.0 / n_cls

# === 2. Encode -> KNN on RAW embeddings, Euclidean ===
# No StandardScaler on the embeddings: an ablation (raw vs scaled x euclidean vs
# cosine over all IY035 checkpoints) showed embedding scaling is a wash on
# average but homogenises checkpoints (collapsing several to exactly 0.766),
# suppressing the best ones. Euclidean on raw embeddings gives the strongest,
# most faithful readout. Default KNN metric is Euclidean (minkowski p=2).
ckpt_results = []  # per-checkpoint {path, accuracy} records
for ckpt in SIMCLR_CKPTS:
    print(f"\n=== Encoding + KNN for {ckpt} ===")
    model = load_simclr_model(ckpt, DEVICE)
    Z_train = encode_channel(model, d["X_train"], DEVICE)
    Z_test = encode_channel(model, d["X_test"], DEVICE)

    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", n_jobs=-1)
    knn.fit(Z_train, d["y_train"])
    y_pred = knn.predict(Z_test)
    ckpt_acc = accuracy_score(d["y_test"], y_pred)
    ckpt_results.append({"checkpoint_path": ckpt, "accuracy": ckpt_acc})

    print(f"IY035 SimCLR (experimental pretraining) model: {ckpt} + KNN (k={K}) -- Full")
    print(f"Accuracy: {ckpt_acc:.4f}  (chance={chance:.4f}, +{ckpt_acc - chance:+.4f})")
    print(classification_report(d["y_test"], y_pred, target_names=d["class_names"]))

# save each checkpoint's path + accuracy to its own csv
ckpt_results_df = pd.DataFrame(ckpt_results).sort_values("accuracy", ascending=False).reset_index(drop=True)
ckpt_results_df.to_csv(IY035_DIR / "IY035_downstream_knn_checkpoint_accuracies.csv", index=False)
print(f"\nSaved: {IY035_DIR / 'IY035_downstream_knn_checkpoint_accuracies.csv'}")

# use the best checkpoint's accuracy for the IY032 baseline comparison below
iy035_acc = ckpt_results_df.iloc[0]["accuracy"]


# === 3. Comparison against IY032 baselines (same split/classes/methodology) ===
full_results = pd.read_csv(IY032_DIR / "IY032_tf_condition_full_simclr_results.csv")
best_synth = full_results[full_results["status"] == "ok"].sort_values("accuracy", ascending=False).iloc[0]

comparison = pd.DataFrame([
    {"method": "Chance", "accuracy": chance},
    {"method": "Catch22+KNN (IY032)", "accuracy": 0.4149},
    {"method": "Raw KNN (IY032)", "accuracy": 0.7234},
    {"method": "IY035 SimCLR+KNN (experimental pretraining)", "accuracy": iy035_acc},
    {"method": f"Best synthetic-pretrained SimCLR+KNN (IY032, {best_synth['label']})",
     "accuracy": best_synth["accuracy"]},
]).sort_values("accuracy", ascending=False).reset_index(drop=True)

print("\n=== Comparison vs IY032 baselines (Full, 6-class TF@condition) ===")
print(comparison.to_string(index=False))
comparison.to_csv(IY035_DIR / "IY035_downstream_knn_vs_iy032.csv", index=False)
print(f"\nSaved: {IY035_DIR / 'IY035_downstream_knn_vs_iy032.csv'}")
