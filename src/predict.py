"""Load a model and predict next-day PM2.5. The one inference path for the
API, monitoring (MAE by zone) and the holdout file.

    python -m src.predict               # demo prediction with @champion
    python -m src.predict --version 1   # same, with registry version 1 (v1)
    python -m src.predict --export      # write the champion to models/model.joblib
"""
import argparse
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd

from src.clean import fix_values
from src.config import load_params, repo_path
from src.features import select_features
from src.mlflow_utils import tracking_uri

ALIAS = "champion"
EXPORT_PATH = "models/model.joblib"


def load_model(ref=ALIAS, params=None):
    """Return the fitted sklearn Pipeline.

    ref: a registry alias ("champion"), a version number (1 or "1" is v1),
    or a path to a .joblib file written by export_champion().
    """
    if str(ref).endswith(".joblib"):
        return joblib.load(repo_path(ref))
    params = params or load_params()
    mlflow.set_tracking_uri(tracking_uri(params))
    name = params["mlflow"]["registered_model_name"]
    ref = str(ref)
    uri = f"models:/{name}/{ref}" if ref.isdigit() else f"models:/{name}@{ref}"
    return mlflow.sklearn.load_model(uri)


def prepare(df, params=None):
    """Raw rows -> model input: the same value fixes as training, never drops rows."""
    params = params or load_params()
    return select_features(fix_values(df, params.get("cleaning", {})), params)


def predict_frame(model, df, params=None):
    """Predictions for every row of df, in order (numpy array)."""
    return model.predict(prepare(df, params))


def predict_one(model, record, params=None):
    """Prediction for one request body (dict with the eight input fields)."""
    return float(predict_frame(model, pd.DataFrame([record]), params)[0])


def export_champion(path=EXPORT_PATH, params=None):
    """Save the @champion Pipeline as joblib so the API image needs no MLflow store."""
    out = Path(repo_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(load_model(ALIAS, params), out)
    return out


SAMPLE = {
    "pm25_today": 120.0,
    "wind_kmph": 8.0,
    "humidity_pct": 65.0,
    "temp_c": 24.0,
    "traffic_index": 7.0,
    "stubble_fires": 2.0,
    "day_type": "weekday",
    "station_zone": "Central",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=ALIAS, help="registry version number or alias (default: champion)")
    ap.add_argument("--export", action="store_true", help=f"write the champion to {EXPORT_PATH}")
    args = ap.parse_args()

    if args.export:
        print(f"Exported @{ALIAS} to {export_champion()}")
        return
    model = load_model(args.version)
    print(pd.DataFrame([SAMPLE]).to_string(index=False))
    print(f"\nPredicted next-day PM2.5 ({args.version}): {predict_one(model, SAMPLE):.2f}")


if __name__ == "__main__":
    main()
