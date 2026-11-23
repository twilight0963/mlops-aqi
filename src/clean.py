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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the AQI dataset by removing duplicates, normalizing categories,
    and dropping leakage columns.
    """
    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Normalize day_type
    if "day_type" in df.columns:
        df["day_type"] = (
            df["day_type"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(DAY_TYPE_MAP)
        )

    # Normalize station_zone
    if "station_zone" in df.columns:
        df["station_zone"] = (
            df["station_zone"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(STATION_ZONE_MAP)
        )

    # Remove data leakage
    df = df.drop(
        columns=["pm25_measured_next_day"],
        errors="ignore",
    )

    return df
