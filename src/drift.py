
import numpy as np
import pandas as pd

from src.clean import clean
from src.config import load_params


def calculate_psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    if expected.empty or actual.empty:
        raise ValueError("Both datasets must contain valid values.")

    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least 2.")

    edges = np.unique(
        np.quantile(expected, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    # The outer bins also include values outside the reference range.
    edges[0], edges[-1] = -np.inf, np.inf
    expected_counts = np.histogram(expected, bins=edges)[0]
    actual_counts = np.histogram(actual, bins=edges)[0]

    eps = 1e-6

    # Smooth each bin and renormalize so the distributions still sum to one.
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def mae_by_group(
    df,
    actual_col,
    predicted_col,
    group_col="station_zone"
):
    required = [actual_col, predicted_col, group_col]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    data = df[required].dropna().copy()

    if data.empty:
        raise ValueError("No valid rows available.")

    data["absolute_error"] = (
        data[actual_col] - data[predicted_col]
    ).abs()

    return (
        data.groupby(group_col)["absolute_error"]
        .mean()
        .rename("mae")
        .reset_index()
    )


def mae_v1_by_zone(df, params=None, version="1"):
    """v1's MAE on df (e.g. later.csv) for each station_zone.

    Loads registry version 1 (v1, trained on cleaned train.csv only) and scores
    it through the same prediction path the API uses.
    """
    from src.predict import load_model, predict_frame
    params = params or load_params()
    data = clean(df, params.get("cleaning", {}))
    data["predicted"] = predict_frame(load_model(version, params), data, params)
    return mae_by_group(data, params["features"]["target"], "predicted", group_col="station_zone")


def feature_psi(reference, current, columns, bins=10):
    """Compute reference-quantile PSI for each numeric feature."""
    return {col: calculate_psi(reference[col], current[col], bins) for col in columns}


def compare_versions(later, v1, v2, params=None):
    """Score both supplied fitted models on the same last 30% of dated rows."""
    from sklearn.metrics import mean_absolute_error
    from src.data import split_later
    from src.predict import predict_frame

    params = params or load_params()
    data = clean(later, params.get("cleaning", {}))
    _, check = split_later(data)
    target = params["features"]["target"]
    if check.empty or check[target].isna().any():
        raise ValueError("Check rows need non-missing targets.")
    mae_v1 = float(mean_absolute_error(check[target], predict_frame(v1, check, params)))
    mae_v2 = float(mean_absolute_error(check[target], predict_frame(v2, check, params)))
    return {"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2,
            "decision": "promote" if mae_v2 <= 0.88 * mae_v1 else "keep"}
