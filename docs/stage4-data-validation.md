# Stage 4 - Data Validation (Abhinav)

## Data Quality Verification
- Rechecked the data cleaning pipeline and data quality issues.
- Confirmed removal of duplicate rows and leakage column.
- Confirmed correction of the Central zone PM2.5 unit issue.
- Confirmed handling of the wind speed sentinel value 9999.

## Drift Monitoring
- Verified PSI between cleaned training data and later data.
- Most drifted feature: stubble_fires.
- PSI: 9.62198.
- This indicates substantial distribution drift and requires monitoring.

## Model Evaluation
- Verified v1 MAE on check data: 63.50823.
- Verified v2 MAE on check data: 40.92461.
- V2 improved MAE by approximately 35.56%.
- Promotion criterion of at least 12% improvement was satisfied.

## Group Performance
- Central MAE: 125.41417.
- Industrial MAE: 24.29281.
- Residential MAE: 23.30605.
- Worst-served group: Central.

## Verification
- Ran python -m src.monitor.
- Ran python -m pytest -q.
- Test result: 35 passed.
