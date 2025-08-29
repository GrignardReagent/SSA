import sys
from pathlib import Path

import pandas as pd

# Ensure the src directory is on the Python path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.data_processing import add_nearest_neighbour_labels


def test_add_nearest_neighbour_labels_pairing():
    df = pd.DataFrame(
        {
            "mu_target": [1.0, 1.1, 2.0, 2.05],
            "cv_target": [0.1, 0.1, 0.2, 0.2],
            "t_ac_target": [0.5, 0.5, 0.6, 0.6],
        }
    )
    labelled = add_nearest_neighbour_labels(df)
    assert labelled["label"].tolist() == [0, 1, 0, 1]

