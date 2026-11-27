"""Holdout writer tests. Synthetic data only (CI has no CSVs or MLflow store)."""
import pytest

from src.predict_holdout import holdout_predictions
from tests.test_predict import PARAMS, fitted_pipeline
from tests.test_train import synthetic


def test_one_row_per_record_in_order_with_two_columns():
    df = synthetic(7, seed=4).drop(columns=["pm25_next_day", "pm25_measured_next_day"])
    df.loc[0, "wind_kmph"] = 9999  # sentinel must not drop the row
    out = holdout_predictions(fitted_pipeline(), df, PARAMS)
    assert list(out.columns) == ["record_id", "prediction"]
    assert out["record_id"].tolist() == df["record_id"].tolist()


def test_duplicate_record_id_is_rejected():
    df = synthetic(3, seed=5)
    df.loc[2, "record_id"] = df.loc[0, "record_id"]
    with pytest.raises(ValueError, match="record_id"):
        holdout_predictions(fitted_pipeline(), df, PARAMS)
