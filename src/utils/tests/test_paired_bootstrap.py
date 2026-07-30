import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from utils.metrics import paired_bootstrap_diff

# small on purpose -- these assert behaviour, not interval precision
N_BOOT = 500


def _labels(n=240, n_classes=6, seed=0):
    return np.random.default_rng(seed).integers(0, n_classes, n)


def test_identical_predictions_give_zero_difference_and_an_interval_spanning_zero():
    y = _labels()
    pred = y.copy()
    pred[::5] = (pred[::5] + 1) % 6          # same errors for both models
    r = paired_bootstrap_diff(y, pred, pred, n_boot=N_BOOT)
    assert r["diff"] == 0.0
    assert r["lo"] == 0.0 and r["hi"] == 0.0  # pairing cancels exactly


def test_strictly_better_model_gives_an_interval_entirely_above_zero():
    y = _labels()
    good = y.copy()
    good[::20] = (good[::20] + 1) % 6        # ~5% wrong
    bad = y.copy()
    bad[::3] = (bad[::3] + 1) % 6            # ~33% wrong
    r = paired_bootstrap_diff(y, good, bad, n_boot=N_BOOT)
    assert r["diff"] > 0
    assert r["lo"] > 0, "a clearly better model must not have an interval crossing zero"
    assert r["p_gt_0"] > 0.99


def test_sign_flips_when_the_arguments_are_swapped():
    y = _labels()
    a, b = y.copy(), y.copy()
    a[::20] = (a[::20] + 1) % 6
    b[::3] = (b[::3] + 1) % 6
    fwd = paired_bootstrap_diff(y, a, b)
    rev = paired_bootstrap_diff(y, b, a)
    assert np.isclose(fwd["diff"], -rev["diff"])
    assert np.isclose(fwd["lo"], -rev["hi"], atol=0.02)


def test_reproducible_under_a_fixed_random_state():
    y = _labels()
    a, b = y.copy(), y.copy()
    a[::7] = (a[::7] + 1) % 6
    b[::9] = (b[::9] + 2) % 6
    r1 = paired_bootstrap_diff(y, a, b, n_boot=N_BOOT, random_state=3)
    r2 = paired_bootstrap_diff(y, a, b, n_boot=N_BOOT, random_state=3)
    r3 = paired_bootstrap_diff(y, a, b, n_boot=N_BOOT, random_state=4)
    assert (r1["lo"], r1["hi"]) == (r2["lo"], r2["hi"])
    assert (r1["lo"], r1["hi"]) != (r3["lo"], r3["hi"])


def test_accepts_an_alternative_metric():
    # the helper must not be hard-wired to balanced accuracy
    y = _labels()
    a, b = y.copy(), y.copy()
    b[::4] = (b[::4] + 1) % 6
    r = paired_bootstrap_diff(y, a, b, metric=accuracy_score, n_boot=N_BOOT)
    assert np.isclose(r["diff"], accuracy_score(y, a) - accuracy_score(y, b))


def test_mismatched_lengths_raise():
    y = _labels(n=100)
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_diff(y, y, y[:50])


def test_skips_resamples_that_lose_a_class():
    # one class with a single member -> many resamples drop it; those must be
    # skipped rather than scored, but enough valid draws must survive
    y = np.array([0] * 120 + [1] * 120 + [2])
    a, b = y.copy(), y.copy()
    b[::10] = (b[::10] + 1) % 3
    r = paired_bootstrap_diff(y, a, b, n_boot=2000)
    assert 0 < r["n_valid"] < 2000
