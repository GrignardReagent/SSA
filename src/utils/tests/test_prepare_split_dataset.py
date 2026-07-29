import numpy as np
import pytest

from utils.processing.pipeline import prepare_split_dataset, resample_to_length


def _toy_dataset(counts=(60, 40, 30), n_tp=20, seed=0):
    """Build a (ts_raw, label_strs) pair with a known, imbalanced class layout.

    Each class lives in its own "file" so the per-file imputation path is
    exercised, mirroring how the real loader returns one array per CSV.
    """
    rng = np.random.default_rng(seed)
    ts_raw, label_strs = [], []
    for i, n in enumerate(counts):
        ts_raw.append(rng.normal(loc=i, size=(n, n_tp)))
        label_strs += [f"class_{i}"] * n
    return ts_raw, label_strs


KEPT = ["class_0", "class_1"]


def test_split_sizes_are_fractions_of_the_whole():
    # 60/20/20 of the 100 cells in the two kept classes -- the val carve must be
    # rescaled by the surviving fraction, or this yields 64/16/20 instead.
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, KEPT, "toy", val_fraction=0.2, test_fraction=0.2)
    assert len(d["y_train"]) == 60
    assert len(d["y_val"]) == 20
    assert len(d["y_test"]) == 20


def test_splits_are_disjoint_and_cover_every_kept_cell():
    # The leakage guarantee this function exists for: no cell may appear in two
    # splits, and none may be silently dropped.
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, KEPT, "toy")
    rows = {name: {r.tobytes() for r in d[f"X_{name}_raw"]}
            for name in ("train", "val", "test")}
    assert rows["train"].isdisjoint(rows["val"])
    assert rows["train"].isdisjoint(rows["test"])
    assert rows["val"].isdisjoint(rows["test"])
    assert sum(len(r) for r in rows.values()) == 100


def test_splits_are_stratified():
    # Class proportions (60/40 here) must be preserved in each split, otherwise
    # a small test split can drift away from the training distribution.
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, KEPT, "toy")
    for name in ("train", "val", "test"):
        y = d[f"y_{name}"]
        assert np.isclose(np.mean(y == 0), 0.6, atol=0.05), name


def test_scaler_is_fit_on_train_only():
    # Train is standardised to ~zero mean/unit variance; val and test are
    # transformed by those same statistics, so they must NOT be exactly centred.
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, KEPT, "toy")
    assert np.allclose(d["X_train"].mean(axis=0), 0, atol=1e-9)
    assert np.allclose(d["X_train"].std(axis=0), 1, atol=1e-9)
    assert not np.allclose(d["X_test"].mean(axis=0), 0, atol=1e-9)
    # the returned scaler must be the one that produced these arrays
    assert np.allclose(d["scaler"].transform(d["X_test_raw"]), d["X_test"])


def test_no_nans_survive_imputation():
    ts, lbl = _toy_dataset()
    ts[0][0, 3] = np.nan
    ts[1][5, :2] = np.nan
    d = prepare_split_dataset(ts, lbl, KEPT, "toy")
    for name in ("train", "val", "test"):
        assert not np.isnan(d[f"X_{name}_raw"]).any(), name


def test_balance_subsamples_to_minority_count():
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, KEPT, "toy", balance=True)
    total = sum(len(d[f"y_{n}"]) for n in ("train", "val", "test"))
    assert total == 80              # 2 classes x min(60, 40)
    y_all = np.concatenate([d[f"y_{n}"] for n in ("train", "val", "test")])
    assert np.bincount(y_all).tolist() == [40, 40]


def test_balance_reproduces_prepare_datasets_balanced_pool():
    # The 3-way split means the two functions cannot be bit-identical end to end,
    # but with balance=True the POOL they split must be the same cells, since
    # both truncate to the shortest kept trace and call balance_classes with the
    # same seed. This is what makes a balance=True run legacy-comparable.
    from utils.processing.pipeline import prepare_dataset

    ts, lbl = _toy_dataset()
    old = prepare_dataset(ts, lbl, KEPT, "old")
    new = prepare_split_dataset(ts, lbl, KEPT, "new", balance=True)
    pool_old = {r.tobytes() for r in np.vstack([old["X_train_raw"], old["X_test_raw"]])}
    pool_new = {r.tobytes() for r in np.vstack(
        [new["X_train_raw"], new["X_val_raw"], new["X_test_raw"]])}
    assert pool_old == pool_new


def test_return_rest_yields_the_non_kept_cells_at_matching_length():
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, KEPT, "toy", return_rest=True)
    assert d["X_rest_raw"].shape == (30, d["min_T"])   # class_2 only
    assert set(d["y_rest_str"]) == {"class_2"}
    # rest cells must not appear in any evaluation split -- the structural
    # guarantee that replaces post-hoc trace-hash exclusion
    rest = {r.tobytes() for r in d["X_rest_raw"]}
    for name in ("train", "val", "test"):
        assert rest.isdisjoint({r.tobytes() for r in d[f"X_{name}_raw"]}), name


def test_rest_is_empty_when_all_classes_are_kept():
    ts, lbl = _toy_dataset()
    d = prepare_split_dataset(ts, lbl, ["class_0", "class_1", "class_2"], "toy",
                              return_rest=True)
    assert d["X_rest_raw"].shape[0] == 0


def test_seq_len_resamples_files_of_differing_lengths():
    # Files of 20 and 12 timepoints: without seq_len everything truncates to 12;
    # with seq_len=20 the short file is upsampled instead of the long one cut.
    rng = np.random.default_rng(1)
    ts = [rng.normal(size=(40, 20)), rng.normal(size=(40, 12))]
    lbl = ["class_0"] * 40 + ["class_1"] * 40
    assert prepare_split_dataset(ts, lbl, KEPT, "toy")["min_T"] == 12
    d = prepare_split_dataset(ts, lbl, KEPT, "toy", seq_len=20)
    assert d["min_T"] == 20
    assert d["X_train_raw"].shape[1] == 20


def test_resample_to_length_endpoints_and_shape():
    row = np.arange(10.0)
    assert resample_to_length(row, 10) is row
    assert resample_to_length(row, 4).tolist() == [0, 1, 2, 3]     # truncate
    up = resample_to_length(row, 19)                               # upsample
    assert len(up) == 19 and up[0] == 0 and up[-1] == 9
    assert set(up).issubset(set(row))   # no interpolated values invented


def test_reproducible_under_the_same_seed():
    ts, lbl = _toy_dataset()
    a = prepare_split_dataset(ts, lbl, KEPT, "toy", random_state=7)
    b = prepare_split_dataset(ts, lbl, KEPT, "toy", random_state=7)
    assert np.array_equal(a["X_test_raw"], b["X_test_raw"])
    c = prepare_split_dataset(ts, lbl, KEPT, "toy", random_state=8)
    assert not np.array_equal(a["X_test_raw"], c["X_test_raw"])


@pytest.mark.parametrize("val,test", [(0.0, 0.0), (0.5, 0.5), (0.9, 0.2), (-0.1, 0.2)])
def test_invalid_fractions_are_rejected(val, test):
    ts, lbl = _toy_dataset()
    with pytest.raises(ValueError):
        prepare_split_dataset(ts, lbl, KEPT, "toy", val_fraction=val, test_fraction=test)


def test_unknown_kept_class_raises():
    ts, lbl = _toy_dataset()
    with pytest.raises(ValueError, match="no cells matched"):
        prepare_split_dataset(ts, lbl, ["nonexistent"], "toy")
