"""Stage 4: train v2, register it, and promote it if the client's rule says so.

    python -m src.retrain

v2 is the v1 configuration (ridge_baseline) refit on cleaned train.csv plus the
first 70% of cleaned later.csv (src.data.split_later). v1 (registry version 1)
and v2 are scored on the remaining 30% with src.drift.compare_versions, and
@champion moves to v2 only on "promote". The champion is then exported to
models/model.joblib for the API.

Idempotent: if a version tagged role=v2 already exists it is reused.
"""
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.clean import clean
from src.config import load_params, repo_path
from src.data import dvc_md5, file_md5, load_train, split_later
from src.drift import compare_versions
from src.features import build_pipeline, make_estimator, select_features
from src.mlflow_utils import setup
from src.predict import ALIAS, export_champion, load_model
from src.train import SKOPS_TRUSTED_TYPES, git_commit

V1_RUN_NAME = "ridge_baseline"
RUN_NAME = "v2_train_plus_later70"
LATER_PATH = "data/later.csv"
LATER_DVC = "data/later.csv.dvc"


def load_later():
    """Original later.csv after checking it matches its .dvc file."""
    path = repo_path(LATER_PATH)
    expected = dvc_md5(repo_path(LATER_DVC))
    if file_md5(path) != expected:
        raise ValueError(f"{path} does not match the md5 in {LATER_DVC}")
    return pd.read_csv(path), expected


def train_v2(params, experiment_id, train, data_info, later, later_md5):
    """Fit v2, log it as one MLflow run and register it. Returns the version."""
    cfg = next(r for r in params["runs"] if r["name"] == V1_RUN_NAME)
    target = params["features"]["target"]
    early, _ = split_later(clean(later, params.get("cleaning", {})))
    rows = pd.concat([train, early], ignore_index=True)
    X, y = select_features(rows, params), rows[target]

    with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
        pipe = build_pipeline(params, make_estimator(cfg["model"], cfg.get("params"), params["seed"]))
        pipe.fit(X, y)
        mlflow.log_params({f"model__{k}": v for k, v in (cfg.get("params") or {}).items()})
        mlflow.log_params({
            "model": cfg["model"],
            "seed": params["seed"],
            "use_clean": params["data"]["use_clean"],
            "data_sha256": data_info["data_sha256"],
            "later_md5": later_md5,
            "later_frac": 0.7,
            "n_train": len(rows),
            "n_train_later": len(early),
            "features": ",".join(X.columns),
        })
        mlflow.set_tags({**data_info, "git_commit": git_commit(), "role": "v2"})
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(X, pipe.predict(X)),
            input_example=X.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[cfg["model"]],
        )
        run_id = run.info.run_id
        model_id = mlflow.last_logged_model().model_id

    mv = mlflow.register_model(f"models:/{model_id}", params["mlflow"]["registered_model_name"])
    MlflowClient().set_model_version_tag(mv.name, mv.version, "role", "v2")
    print(f"Logged run {run_id} and registered it as {mv.name} v{mv.version}.")
    return mv.version


def main(params=None):
    params = params or load_params()
    experiment_id = setup(params)
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    train, data_info = load_train(params)  # verifies train.csv against its .dvc
    later, later_md5 = load_later()

    existing = [v for v in client.search_model_versions(f"name = '{name}'")
                if v.tags.get("role") == "v2"]
    if existing:
        version = existing[0].version
        print(f"v2 already registered as {name} v{version}; reusing it.")
    else:
        version = train_v2(params, experiment_id, train, data_info, later, later_md5)

    result = compare_versions(later, load_model("1", params), load_model(version, params), params)
    with mlflow.start_run(run_id=client.get_model_version(name, version).run_id):
        mlflow.log_metrics({k: v for k, v in result.items() if k.startswith("mae")})
    print(f"MAE on last 30% of later.csv: v1 {result['mae_v1_on_check']:.4f}  "
          f"v2 {result['mae_v2_on_check']:.4f}  -> {result['decision']}")

    if result["decision"] == "promote":
        client.set_registered_model_alias(name, ALIAS, version)
        print(f"@{ALIAS} -> v{version}")
    print(f"Exported @{ALIAS} to {export_champion(params=params)}")
    return version, result


if __name__ == "__main__":
    main()
