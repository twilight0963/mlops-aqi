"""Register the best logged run (lowest validation MAE) and point @champion at it.

    python -m src.register                  # best run trained on clean() data
    python -m src.register --allow-baseline # also consider runs on raw data

Idempotent: if the best run is already registered, nothing new is created.
"""
import argparse

import mlflow
from mlflow import MlflowClient

from src.config import load_params
from src.mlflow_utils import setup

ALIAS = "champion"


def best_run(experiment_id, allow_baseline):
    filt = "attributes.status = 'FINISHED'"  # never pick a crashed run without a model
    if not allow_baseline:
        filt += " and params.use_clean = 'True'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filt,
        order_by=["metrics.mae ASC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(
            "No eligible runs. Train on cleaned data first (use_clean: true), "
            "or pass --allow-baseline to register a raw-data run."
        )
    return runs[0]


def register(params, allow_baseline=False):
    experiment_id = setup(params)
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    run = best_run(experiment_id, allow_baseline)
    run_id = run.info.run_id

    existing = [v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run_id]
    if existing:
        version = existing[0].version
        print(f"Run {run_id} already registered as {name} v{version}; not creating a new version.")
    else:
        # mv = mlflow.register_model(f"runs:/{run_id}/model", name)
        logged_models = client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
            max_results=10,
            )
        if not logged_models:
            raise RuntimeError(f"No logged model found for run {run_id}")
        logged_model = next(
            (m for m in logged_models if m.status == "READY"),
            None,
        )
        if logged_model is None:
            raise RuntimeError(f"No READY model found for run {run_id}")
        mv = mlflow.register_model(f"models:/{logged_model.model_id}", name)
        version = mv.version
        client.set_model_version_tag(name, version, "mae", f"{run.data.metrics['mae']:.4f}")
        client.set_model_version_tag(name, version, "use_clean", run.data.params.get("use_clean"))
        print(f"Registered run {run_id} as {name} v{version}.")

    client.set_registered_model_alias(name, ALIAS, version)
    print(f"@{ALIAS} -> v{version}  (val MAE {run.data.metrics['mae']:.2f}, "
          f"run '{run.info.run_name}', use_clean={run.data.params.get('use_clean')})")
    print(f"Load with: mlflow.pyfunc.load_model('models:/{name}@{ALIAS}')")
    return run_id, version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-baseline", action="store_true",
                    help="consider runs trained without clean() as well")
    args = ap.parse_args()
    register(load_params(), allow_baseline=args.allow_baseline)


if __name__ == "__main__":
    main()
