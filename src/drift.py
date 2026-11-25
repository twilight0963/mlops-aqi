
import numpy as np
import pandas as pd

from src.clean import clean
from src.config import load_params
from src.predict import load_model, predict_frame


def calculate_psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    if expected.empty or actual.empty:
        raise ValueError("Both datasets must contain valid values.")

    if bins < 1:
        raise ValueError("bins must be at least 1.")

    edges = np.unique(
        np.quantile(expected, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=edges)[0]
    actual_counts = np.histogram(actual, bins=edges)[0]

    expected_counts[0] += np.sum(expected < edges[0])
    expected_counts[-1] += np.sum(expected > edges[-1])

    actual_counts[0] += np.sum(actual < edges[0])
    actual_counts[-1] += np.sum(actual > edges[-1])

    eps = 1e-6

    expected_pct = np.clip(expected_counts / len(expected), eps, None)
    actual_pct = np.clip(actual_counts / len(actual), eps, None)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def mae_by_group(
    df,
    actual_col,
    predicted_col,
    group_col="zone"
):
    required = [actual_col, predicted_col, group_col]

    if not all(col in df.columns for col in required):
        raise ValueError("Required columns are missing.")

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
    params = params or load_params()
    data = clean(df, params.get("cleaning", {}))
    data["predicted"] = predict_frame(load_model(version, params), data, params)
    return mae_by_group(data, params["features"]["target"], "predicted", group_col="station_zone")
