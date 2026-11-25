"""Prediction helper tests. No MLflow store or DVC data needed (CI has neither)."""
import joblib
import numpy as np

from src.clean import fix_values
from src.config import load_params
from src.features import build_pipeline, make_estimator, select_features
from src.predict import load_model, predict_frame, predict_one
from tests.test_train import synthetic

PARAMS = load_params()


def fitted_pipeline():
    df = synthetic()
    return build_pipeline(PARAMS, make_estimator("ridge", {}, 0)).fit(
        select_features(df, PARAMS), df["pm25_next_day"])


def request(**overrides):
    body = {
        "pm25_today": 120.0, "wind_kmph": 8.0, "humidity_pct": 65, "temp_c": 24.0,
        "traffic_index": 7, "stubble_fires": 2, "day_type": "weekday", "station_zone": "Central",
    }
    return {**body, **overrides}


def test_fix_values_keeps_duplicate_rows():
    df = synthetic(3)
    doubled = df.iloc[[0, 0, 1]]
    assert len(fix_values(doubled, PARAMS["cleaning"])) == 3


def test_predict_one_applies_training_fixes():
    model = fitted_pipeline()
    clean_pred = predict_one(model, request(), PARAMS)
    messy = request(day_type=" WEEKDAY ", station_zone="central", pm25_today=120_000.0)
    assert predict_one(model, messy, PARAMS) == clean_pred  # case + unit mix-up fixed


def test_predict_handles_sentinel_and_returns_one_value_per_row():
    model = fitted_pipeline()
    df = synthetic(5, seed=2)
    df.loc[0, "wind_kmph"] = 9999
    preds = predict_frame(model, df, PARAMS)
    assert preds.shape == (5,) and np.isfinite(preds).all()
    assert isinstance(predict_one(model, request(wind_kmph=9999), PARAMS), float)


def test_joblib_export_round_trip(tmp_path):
    model = fitted_pipeline()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    df = synthetic(5, seed=3)
    np.testing.assert_array_equal(
        predict_frame(load_model(str(path)), df, PARAMS), predict_frame(model, df, PARAMS))
