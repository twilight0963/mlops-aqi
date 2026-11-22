"""Train every run configured in params.yaml and log each to MLflow.

Run from anywhere:  python -m src.train
Registering a model is a separate step (python -m src.register) so that
re-running training never silently creates registry versions.
"""
import random
import subprocess

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

from src.config import REPO_ROOT, load_params
from src.data import load_train, time_split
from src.features import build_pipeline, make_estimator, select_features
from src.mlflow_utils import setup

# Types skops must be told to trust when saving each model (we create these files
# ourselves, so they are safe). Found with skops.io.get_untrusted_types().
SKOPS_TRUSTED_TYPES = {
    "ridge": ["numpy.dtype"],
    "random_forest": ["numpy.dtype", "sklearn.tree._tree.Tree"],
    "hist_gb": ["numpy.dtype", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_data(params):
    """Load (DVC-verified) train.csv, optionally clean it, and split by date."""
    df, data_info = load_train(params)
    if params["data"]["use_clean"]:
        from src.clean import clean  # imported only when enabled (Abhinav's module)
        df = clean(df)
    s = params["split"]
    train_df, val_df = time_split(df, s["date_col"], s["test_size"])
    return train_df, val_df, data_info


def main(params=None):
    params = params or load_params()
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)

    train_df, val_df, data_info = prepare_data(params)
    target = params["features"]["target"]
    X_train, y_train = select_features(train_df, params), train_df[target]
    X_val, y_val = select_features(val_df, params), val_df[target]

    # Reference point: predicting the training mean for every row.
    mean_mae = mean_absolute_error(y_val, np.full(len(y_val), y_train.mean()))

    experiment_id = setup(params)
    commit = git_commit()
    results = []
    for cfg in params["runs"]:
        with mlflow.start_run(experiment_id=experiment_id, run_name=cfg["name"]) as run:
            pipe = build_pipeline(params, make_estimator(cfg["model"], cfg.get("params"), seed))
            pipe.fit(X_train, y_train)  # imputer/encoder/scaler see train rows only
            mae = mean_absolute_error(y_val, pipe.predict(X_val))
            train_mae = mean_absolute_error(y_train, pipe.predict(X_train))

            mlflow.log_params({f"model__{k}": v for k, v in (cfg.get("params") or {}).items()})
            mlflow.log_params({
                "model": cfg["model"],
                "seed": seed,
                "use_clean": params["data"]["use_clean"],
                "split_method": params["split"]["method"],
                "test_size": params["split"]["test_size"],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "features": ",".join(X_train.columns),
            })
            mlflow.set_tags({**data_info, "git_commit": commit, "run_id": run.info.run_id})
            mlflow.log_metrics({"mae": mae, "train_mae": train_mae, "baseline_mean_mae": mean_mae})
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=infer_signature(X_val, pipe.predict(X_val)),
                input_example=X_val.head(3),
                skops_trusted_types=SKOPS_TRUSTED_TYPES[cfg["model"]],  # skops, not pickle
            )
            results.append((cfg["name"], run.info.run_id, mae, train_mae))

    print(f"\nExperiment: {params['mlflow']['experiment_name']}   use_clean={params['data']['use_clean']}")
    print(f"data_sha256: {data_info['data_sha256']}")
    print(f"mean-predictor val MAE: {mean_mae:.2f}\n")
    print(f"{'run':<16}{'run_id':<34}{'val_mae':>9}{'train_mae':>11}")
    for name, rid, mae, tmae in sorted(results, key=lambda r: r[2]):
        print(f"{name:<16}{rid:<34}{mae:>9.2f}{tmae:>11.2f}")
    return results


if __name__ == "__main__":
    main()
