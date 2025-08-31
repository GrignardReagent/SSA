import sys
from pathlib import Path

import pandas as pd

# Ensure the src directory is on the Python path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.data_processing import add_labels


def test_add_labels_nearest_neighbour_pairing():
    df = pd.DataFrame(
        {
            "mu_target": [1.0, 1.1, 2.0, 2.05],
            "cv_target": [0.1, 0.1, 0.2, 0.2],
            "t_ac_target": [0.5, 0.5, 0.6, 0.6],
        }
    )
    labelled = add_labels(df, labeling_regime="nearest_neighbour")
    assert labelled["label"].tolist() == [0, 1, 0, 1]


def test_add_labels_binary_split():
    df = pd.DataFrame({"mu_target": [1.0, 2.0, 3.0, 4.0]})
    labelled = add_labels(df, labeling_regime="binary", column="mu_target")
    assert labelled["label"].tolist() == [0, 0, 1, 1]

