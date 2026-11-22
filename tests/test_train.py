import numpy as np
import pandas as pd
import pytest

from src.config import load_params
from src.data import split_later
from src.features import build_pipeline, feature_columns, make_estimator, select_features

PARAMS = load_params()


def synthetic(n=60, seed=0):
    """Small frame shaped like train.csv, built in the test (CI has no DVC data)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "pm25_today": rng.uniform(20, 300, n),
        "wind_kmph": rng.uniform(0, 30, n),
        "humidity_pct": rng.integers(20, 95, n),
        "temp_c": rng.uniform(5, 35, n),
        "traffic_index": rng.integers(10, 100, n),
        "stubble_fires": rng.integers(0, 700, n),
        "day_type": rng.choice(["weekday", "weekend", "festival"], n),
        "station_zone": rng.choice(["Central", "Residential", "Industrial"], n),
    })
    df["pm25_measured_next_day"] = df["pm25_today"] * 1.1
    df["pm25_next_day"] = df["pm25_measured_next_day"] + rng.normal(0, 5, n)
    return df


def test_features_exclude_leak_id_and_target():
    cols = feature_columns(PARAMS)
    assert len(cols) == 8
    for bad in ["pm25_measured_next_day", "record_id", "date", "pm25_next_day"]:
        assert bad not in cols


def test_feature_columns_rejects_leak():
    bad = {**PARAMS, "features": {**PARAMS["features"],
                                  "numeric": PARAMS["features"]["numeric"] + ["pm25_measured_next_day"]}}
    with pytest.raises(ValueError):
        feature_columns(bad)


@pytest.mark.parametrize("model", ["ridge", "random_forest", "hist_gb"])
def test_pipeline_handles_nans_and_unseen_category(model):
    df = synthetic()
    df.loc[0:4, "wind_kmph"] = np.nan
    df.loc[5:7, "station_zone"] = np.nan
    pipe = build_pipeline(PARAMS, make_estimator(model, {}, seed=1))
    pipe.fit(select_features(df, PARAMS), df["pm25_next_day"])

    new = synthetic(5, seed=1)
    new.loc[0, "station_zone"] = "Riverside"   # never seen in training
    new.loc[1, "humidity_pct"] = np.nan
    preds = pipe.predict(select_features(new, PARAMS))
    assert preds.shape == (5,) and np.isfinite(preds).all()


def test_same_seed_gives_identical_predictions():
    df = synthetic()
    X, y = select_features(df, PARAMS), df["pm25_next_day"]
    p1 = build_pipeline(PARAMS, make_estimator("random_forest", {"n_estimators": 20}, 7)).fit(X, y).predict(X)
    p2 = build_pipeline(PARAMS, make_estimator("random_forest", {"n_estimators": 20}, 7)).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)


def test_unknown_model_name_rejected():
    with pytest.raises(ValueError):
        make_estimator("xgboost", {}, 0)


def test_split_later_is_by_date_and_deterministic():
    df = synthetic(30).sample(frac=1, random_state=3)  # shuffled input
    first, last = split_later(df, 0.7)
    assert (len(first), len(last)) == (21, 9)
    assert first["date"].max() <= last["date"].min()
    again_first, _ = split_later(df.sample(frac=1, random_state=4), 0.7)
    assert list(first["record_id"]) == list(again_first["record_id"])
