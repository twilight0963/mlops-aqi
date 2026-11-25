"""Data cleaning utilities for the AQI prediction project."""

import pandas as pd


DAY_TYPE_MAP = {
    "weekday": "weekday",
    "weekend": "weekend",
    "festival": "festival",
}

STATION_ZONE_MAP = {
    "industrial": "Industrial",
    "central": "Central",
    "residential": "Residential",
}


def clean(df: pd.DataFrame, rules=None) -> pd.DataFrame:
    """Clean training data using the rules configured in params.yaml."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean() expects a pandas DataFrame")

    # 1. Remove duplicate rows.
    result = df.drop_duplicates().copy()

    # 2-4. Fix values row by row (shared with prediction).
    result = fix_values(result, rules)

    # 5. Drop the target-leaking column.
    return result.drop(columns=["pm25_measured_next_day"], errors="ignore")


def fix_values(df: pd.DataFrame, rules=None) -> pd.DataFrame:
    """Row-wise fixes used for both training and prediction. Never drops rows."""

    rules = rules or {}
    result = df.copy(deep=True)

    # 2. Normalize category spelling and whitespace.
    if "day_type" in result.columns:
        result["day_type"] = (
            result["day_type"].astype("string").str.strip().str.lower()
        )

    if "station_zone" in result.columns:
        result["station_zone"] = (
            result["station_zone"].astype("string").str.strip().str.lower()
            .map(STATION_ZONE_MAP)
            .fillna(result["station_zone"].astype("string").str.strip())
        )

    # 3. Correct the Central-zone PM2.5 unit mix-up.
    unit = rules.get("unit_mixup", {})
    group_col = unit.get("group_column", "station_zone")
    group_value = str(unit.get("group_value", "central")).strip().lower()
    column = unit.get("column", "pm25_today")
    threshold = unit.get("wrong_value_min", 1000)
    factor = unit.get("factor", 1000)

    if group_col in result.columns and column in result.columns:
        affected = (
            result[group_col].astype("string").str.strip().str.lower().eq(group_value)
            & result[column].gt(threshold)
        )
        result.loc[affected, column] = result.loc[affected, column] / factor

    # 4. Replace known sentinel values with NaN.
    for col, values in rules.get("sentinels", {"wind_kmph": [9999]}).items():
        if col in result.columns:
            result[col] = result[col].replace(values, float("nan"))

    return result
