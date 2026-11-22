"""Feature selection and the sklearn Pipeline (preprocessing + model)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ESTIMATORS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "hist_gb": HistGradientBoostingRegressor,
}


def feature_columns(params):
    """The model's input columns, in a fixed order."""
    f = params["features"]
    cols = list(f["numeric"]) + list(f["categorical"])
    forbidden = set(f["drop"]) | {f["target"]}
    leaked = forbidden.intersection(cols)
    if leaked:
        raise ValueError(f"Forbidden columns configured as features: {sorted(leaked)}")
    return cols


def select_features(df, params):
    """Return X with only the configured features (numeric cast to float)."""
    X = df[feature_columns(params)].copy()
    num = params["features"]["numeric"]
    X[num] = X[num].astype(float)  # NaN-safe dtype, stable MLflow signature
    return X


def make_estimator(name, model_params, seed):
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(ESTIMATORS)}.")
    est = ESTIMATORS[name](**(model_params or {}))
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est


def build_pipeline(params, estimator):
    """Imputation + scaling/encoding + model. Fit it on the train split only."""
    f = params["features"]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, list(f["numeric"])),
        ("cat", categorical, list(f["categorical"])),
    ])
    return Pipeline([("preprocess", pre), ("model", estimator)])
