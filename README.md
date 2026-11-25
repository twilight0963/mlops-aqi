# MP221: Tomorrow's Air Quality (Guwahati)

MLOps mini project (SCSE3040), Group G008.

The project predicts next-day PM2.5 levels for the Guwahati Clean Air Cell using machine learning and an end-to-end MLOps pipeline.

## Team

| Member | Role |
|---|---|
| Abhinav | Data Lead |
| Daksh | Model Lead |
| Gauri | Service Lead |
| Sania | Ops Lead |

## Repository Layout

- `src/` — Data cleaning, feature engineering, model training, prediction, API and drift detection code.
- `tests/` — Automated tests using pytest.
- `data/` — Project datasets, with the original `train.csv` tracked using DVC.
- `k8s/` — Kubernetes deployment and service manifests.
- `.github/workflows/` — CI configuration and workflows.
- `predictions/` — Holdout dataset predictions.
- `findings.yaml` — Completed findings for the marking tool.
- `params.yaml` — Central configuration for data, features, training and model settings.

## Data Preparation and Cleaning

The data preparation pipeline is implemented in `src/clean.py`. It prepares the datasets before model training and prediction through the following operations:

- Removes duplicate records.
- Normalizes categorical values.
- Corrects the PM2.5 unit inconsistency in the Central station zone.
- Handles invalid wind-speed sentinel values (`9999`) by converting them into missing values.
- Removes the `pm25_measured_next_day` column to prevent target leakage.

The cleaning function is integrated with the training data loader through the `data.use_clean` setting in `params.yaml`.

### Dataset Cleaning Verification

The cleaning pipeline was tested on all three project datasets.

| Dataset | Original Rows | Cleaned Rows | Missing Values After Cleaning |
|---|---:|---:|---:|
| `train.csv` | 1,058 | 1,000 | 38 |
| `later.csv` | 300 | 300 | 0 |
| `holdout_features.csv` | 250 | 250 | 0 |

The training dataset was reduced from 1,058 to 1,000 rows after duplicate removal. The later and holdout datasets retained all their original records.

### Feature Selection

The `select_features()` function in `src/features.py` ensures that the same eight input features are selected consistently across the training, monitoring and holdout datasets.

The eight model input features are:

1. `pm25_today`
2. `wind_kmph`
3. `humidity_pct`
4. `temp_c`
5. `traffic_index`
6. `stubble_fires`
7. `day_type`
8. `station_zone`

The `record_id` and `date` columns are not passed as model input features. The holdout dataset does not contain the target column and is reserved for final prediction.

## How to Run

Use Python 3.11 (the same version used in CI). Run all commands from the repository root.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run tests

```bash
python -m pytest -v
```

**Note:** The original `data/train.csv` is tracked using DVC. Training verifies its integrity against the corresponding DVC metadata. The original dataset should not be modified.

## Training and Model Registry

All project settings, including the random seed, chronological split, features, MLflow configuration and model parameters, are maintained centrally in `params.yaml`.

### Train models

```bash
python -m src.train
```

This executes one MLflow run for each configuration defined under `runs` in `params.yaml`.

### Register the best model

```bash
python -m src.register
```

The registration script selects the best eligible run based on the lowest validation MAE and registers the model with the `@champion` alias.

### View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the following address in your browser:

`http://127.0.0.1:5000`

### Training and Experiment Details

- **Validation strategy:** Chronological validation, with approximately the latest 20% of dates in `train.csv` reserved for validation.
- **Preprocessing:** Imputation, scaling and one-hot encoding of `day_type` and `station_zone` are implemented as part of the Scikit-learn Pipeline.
- **Data integrity:** Each MLflow run logs the `data_sha256` hash of the original training dataset, verified against the DVC metadata.
- **Experiment tracking:** Runs record

## Loading the Model and Predicting

`src/predict.py` is the single inference path. It applies the same value fixes as training (`src.clean.fix_values`) and `select_features()`, and it never drops rows.

```python
from src.predict import load_model, predict_one, predict_frame

model = load_model()                      # models:/mp221-pm25-model@champion
v1 = load_model("1")                      # registry version 1 (v1)
api_model = load_model("models/model.joblib")  # exported champion, no MLflow store needed

predict_one(model, request_body_dict)     # float, for POST /predict
predict_frame(model, df)                  # numpy array, one prediction per row
```

```bash
python -m src.predict              # demo prediction with @champion
python -m src.predict --version 1  # demo with v1
python -m src.predict --export     # write @champion to models/model.joblib (the API loads this)
```

v1's MAE per `station_zone` on `later.csv`:

```python
import pandas as pd
from src.drift import mae_v1_by_zone
mae_v1_by_zone(pd.read_csv("data/later.csv"))
```

`mlflow.db` is committed; `mlruns/` is not. MLflow stores absolute artifact paths, so registry loading only works on the machine that trained the model. Everywhere else (Docker, CI), use `models/model.joblib`.
