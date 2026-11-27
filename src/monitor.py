"""Reproduce the v1 Ridge run and evaluate a later-data v2 candidate.

Run from the repository root: python -m src.monitor
The check portion of later.csv is never used in either model's training.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.clean import clean
from src.config import load_params, repo_path
from src.data import file_md5, load_train, split_later, time_split, dvc_md5
from src.drift import feature_psi, mae_by_group
from src.features import build_pipeline, make_estimator, select_features


def evaluate():
    params = load_params()
    train, _ = load_train(params)  # Verifies the original train.csv DVC hash.
    later_path = repo_path("data/later.csv")
    later_dvc = repo_path("data/later.csv.dvc")
    if file_md5(later_path) != dvc_md5(later_dvc):
        raise ValueError("later.csv differs from its DVC metadata")
    later = clean(pd.read_csv(later_path), params.get("cleaning", {}))
    early, check = split_later(later)
    target = params["features"]["target"]

    def fit(rows):
        model = build_pipeline(params, make_estimator("ridge", {"alpha": 1.0}, params["seed"]))
        model.fit(select_features(rows, params), rows[target])
        return model

    v1_train, validation = time_split(train, params["split"]["date_col"], params["split"]["test_size"])
    v1 = fit(v1_train)
    val_mae = mean_absolute_error(validation[target], v1.predict(select_features(validation, params)))
    if not np.isclose(val_mae, 8.843130576342366, atol=1e-7):
        raise ValueError("Reproduced v1 does not match the recorded validation MAE")
    exported = joblib.load(repo_path("models/model.joblib"))
    if not np.allclose(v1.predict(select_features(validation, params)),
                       exported.predict(select_features(validation, params)), atol=1e-7):
        raise ValueError("Exported model does not match reproduced v1")

    # Documented local v2 choice: same Ridge pipeline, retrained using all
    # cleaned train rows and only the early 70% of later rows.
    v2 = fit(pd.concat([train, early], ignore_index=True))
    check_x = select_features(check, params)
    mae_v1 = float(mean_absolute_error(check[target], v1.predict(check_x)))
    mae_v2 = float(mean_absolute_error(check[target], v2.predict(check_x)))
    zones = later.copy()
    zones["predicted"] = v1.predict(select_features(later, params))
    group_mae = mae_by_group(zones, target, "predicted").set_index("station_zone")["mae"].to_dict()
    psi = feature_psi(train, later, params["features"]["numeric"])
    return {
        "method": "reproduced Ridge v1; Ridge v2 fit on cleaned train plus first 70% of later",
        "n_v1_train": len(v1_train), "n_v2_train": len(train) + len(early),
        "n_check": len(check), "v1_validation_mae": float(val_mae),
        "psi": psi, "most_drifted_feature": max(psi, key=psi.get),
        "mae_by_group": group_mae,
        "worst_served_group": max(group_mae, key=group_mae.get),
        "mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2,
        "decision": "promote" if mae_v2 <= 0.88 * mae_v1 else "keep",
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
