# MP221: Tomorrow's Air Quality (Guwahati)

MLOps mini project (SCSE3040), group G008.
Predicts next-day PM2.5 for the Guwahati Clean Air Cell.

## Team
| Member | Role |
|---|---|
| Abhinav | Data lead |
| Daksh | Model lead |
| Gauri | Service lead |
| Sania | Ops lead |

## Repository layout
- `src/` : cleaning, training, API, drift code
- `tests/` : pytest tests
- `data/` : datasets (train.csv tracked with DVC)
- `k8s/` : Kubernetes manifests
- `.github/workflows/` : CI
- `predictions/` : holdout predictions
- `findings.yaml` : filled-in findings for the marking tool

## How to run
Use Python 3.11 (same as CI) to avoid errors. Run commands from the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# data/train.csv is tracked by DVC; training aborts if its md5 differs from data/train.csv.dvc
python -m pytest -v
```

### Training and model registry
All settings (seed, split, features, MLflow names, model configs) live in `params.yaml`.

```bash
python -m src.train      # one MLflow run per entry in params.yaml `runs`
python -m src.register   # best run (lowest validation MAE) -> registry version + @champion alias
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs by opening URL http://127.0.0.1:5000
```

- Validation is chronological: the latest ~20% of dates in `train.csv` are held out.
- Preprocessing (imputation, scaling, one-hot encoding of `day_type` / `station_zone`) is part of the
  sklearn Pipeline and is fitted on the training split only.
- Each run logs `data_sha256` (hash of the original `train.csv`, verified against the `.dvc` md5),
  the `mae` metric, hyperparameters and the git commit.
- `register` only considers runs trained on `clean()` data (Set `data.use_clean: true`) unless
  `--allow-baseline` is passed. Re-running it does not create duplicate versions.
- The model loads with `mlflow.pyfunc.load_model("models:/mp221-pm25-model@champion")`; pass inputs
  through `src.features.select_features()` so columns and dtypes match the logged signature.
