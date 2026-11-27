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
- `findings.yaml` — Findings for the marking tool.
- `params.yaml` — Central configuration for data, features, training and model settings.

## Data Preparation and Cleaning

The data preparation pipeline is implemented in `src/clean.py`. It prepares datasets before model training and prediction through the following operations:

- Removes duplicate records.
- Normalizes categorical values.
- Corrects the PM2.5 unit inconsistency in the Central station zone.
- Handles invalid wind-speed sentinel values (`9999`) by converting them into missing values.
- Removes the `pm25_measured_next_day` column to prevent target leakage.

The cleaning function is integrated with the training data loader through the `data.use_clean` setting in `params.yaml`.

### Dataset Cleaning Verification

| Dataset | Original Rows | Cleaned Rows | Missing Values After Cleaning |
|---|---:|---:|---:|
| `train.csv` | 1,058 | 1,000 | 38 |
| `later.csv` | 300 | 300 | 0 |
| `holdout_features.csv` | 250 | 250 | 0 |

The training dataset was reduced from 1,058 to 1,000 rows after duplicate removal. The later and holdout datasets retained their original records.

### Feature Selection

The `select_features()` function in `src/features.py` ensures that the same eight input features are selected consistently across training, monitoring and holdout datasets.

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

Use Python 3.11 (the same version used in CI). Run commands from the repository root.

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

Project settings, including the random seed, chronological split, features, MLflow configuration and model parameters, are maintained centrally in `params.yaml`.

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

Open `http://127.0.0.1:5000` in your browser.

### Training and Experiment Details

- **Validation strategy:** Chronological validation, with approximately the latest 20% of dates reserved for validation.
- **Preprocessing:** Imputation, scaling and one-hot encoding of categorical features are implemented in the Scikit-learn Pipeline.
- **Data integrity:** Training runs record the SHA-256 hash of the original training dataset.
- **Experiment tracking:** MLflow records model parameters, validation MAE, training MAE and dataset information.
- **Model selection:** The model with the lowest validation MAE is selected.
- **Model registry:** The selected model is registered as `mp221-pm25-model` with the `champion` alias.

## Loading the Model and Predicting

`src/predict.py` provides the inference functions. It applies the same value fixes as training and uses `select_features()` to maintain consistent model inputs.

```python
from src.predict import load_model, predict_one, predict_frame

model = load_model()
prediction = predict_one(model, request_body_dict)
predictions = predict_frame(model, df)
```

Export the champion model for portable serving:

```bash
python -m src.predict --export
```

The exported model is saved to `models/model.joblib`.

## Model Serving with FastAPI

The API is implemented in `src/api.py` using FastAPI and Pydantic for request validation.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns API information |
| GET | `/health` | Checks application and model status |
| POST | `/predict` | Predicts next-day PM2.5 |

### Run the API locally

Activate the virtual environment:

```bash
source .venv/bin/activate
```

If Docker is already running on port 8000, stop the container before starting the local server:

```bash
docker compose down
```

Start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to access the interactive Swagger interface.

### Example prediction request

Send the following JSON to `POST /predict`:

```json
{
  "pm25_today": 120.0,
  "wind_kmph": 8.0,
  "humidity_pct": 65.0,
  "temp_c": 24.0,
  "traffic_index": 7.0,
  "stubble_fires": 2.0,
  "day_type": "weekday",
  "station_zone": "Central"
}
```

The API validates the input and returns a next-day PM2.5 prediction.

## Docker Deployment

The application is containerized using Docker and Docker Compose.

### Build and start the API

```bash
docker compose up --build -d
```

### Check container status

```bash
docker compose ps
```

### View API logs

```bash
docker compose logs api
```

The API is available at:

- API: `http://localhost:8000`
- Swagger documentation: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/health`

### Stop the container

```bash
docker compose down
```

The Docker image loads the exported model from `models/model.joblib`. This allows the API container to serve predictions without depending on the local MLflow tracking database.

## Automated Testing

Run the complete test suite:

```bash
python -m pytest -v
```

The currently verified test result is **31 passed**.

The tests cover data processing, feature engineering, model training and API functionality.

## Monitoring and operations

`src/drift.py` computes PSI from ten quantile bins of cleaned training data and
per-zone MAE from predictions and actual next-day PM2.5. Clean `later.csv` with
the same rules before comparing distributions. PSI is a distribution shift
measure, not a model quality score. `compare_versions(later, v1, v2)` orders
rows by date and record ID, scores both supplied fitted models on the last 30%,
and promotes only when v2 MAE is at most 88% of v1 MAE. Use the same held-out
rows for both models; do not train v2 on those rows.

The `k8s/` manifests deploy four replicas of `mp221-api:1.0` with `/health`
readiness and liveness probes and expose service `mp221-svc` on port 80.
Build and make that image available to the cluster before applying the files.
The API emits JSON event logs to standard error; prediction requests do not
log personal data or feature values.

The CSV files are tracked with DVC metadata and excluded from Git. To run
training or monitoring, supply the original datasets at `data/train.csv` and
`data/later.csv` and verify their hashes against the `.dvc` files. CI runs
the unit tests without those data files.

The six PSI values in `findings.yaml` use the supplied DVC-matching train and
later CSVs after `clean()`, with ten reference quantile bins. The checked-in
model export does not establish which registry version it represents, so the
v1/v2 comparison and per-version MAE fields remain unset until both versioned
models are available and evaluated on the same check rows.

The API was also verified through Swagger UI:

- `GET /health` returned HTTP 200 with the model loaded.
- `POST /predict` returned HTTP 200 with a PM2.5 prediction.

## MLflow Artifact Portability

MLflow tracking databases and artifact paths may contain machine-specific information.

The local `mlflow.db` and `mlruns/` directory should not be committed to Git.

For portable model serving, use the exported `models/model.joblib` artifact. The Docker API loads this file directly and does not depend on the local MLflow registry.
