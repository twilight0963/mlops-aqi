"""Stage 4: predict every holdout row with the @champion model.

    python -m src.predict_holdout

Writes predictions/holdout_predictions.csv with exactly record_id,prediction.
"""
import numpy as np
import pandas as pd

from src.config import load_params, repo_path
from src.predict import ALIAS, load_model, predict_frame

HOLDOUT_PATH = "data/holdout_features.csv"
OUTPUT_PATH = "predictions/holdout_predictions.csv"


def holdout_predictions(model, df, params=None):
    """One prediction per input row, in input order, as record_id,prediction."""
    ids = df["record_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("record_id must be present and unique for every holdout row")
    preds = predict_frame(model, df, params)
    if not np.isfinite(preds).all():
        raise ValueError("Model returned non-finite predictions")
    return pd.DataFrame({"record_id": ids.to_numpy(), "prediction": preds})


def main(params=None):
    params = params or load_params()
    holdout = pd.read_csv(repo_path(HOLDOUT_PATH))
    out = holdout_predictions(load_model(ALIAS, params), holdout, params)
    path = repo_path(OUTPUT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} predictions with @{ALIAS} to {path}")


if __name__ == "__main__":
    main()
