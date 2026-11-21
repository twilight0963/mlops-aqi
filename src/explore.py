"""Stage 1 exploration: look for the seven data problems in data/train.csv.

READ ONLY. This script never writes to train.csv.
Run from the repo root:  python src/explore.py
These are hints to confirm by eye. Stage 2 turns each one into a detector in clean.py.
"""
from pathlib import Path

import pandas as pd

TRAIN = Path("data/train.csv")
HOLDOUT = Path("data/holdout_features.csv")
TARGET = "pm25_next_day"
NUMERIC = ["pm25_today", "wind_kmph", "humidity_pct", "temp_c",
           "traffic_index", "stubble_fires"]


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def longest_run(df, col):
    """Longest stretch of identical consecutive values (df must be sorted by date)."""
    s = df[col]
    run_id = (s != s.shift()).cumsum()
    sizes = run_id.value_counts()
    best = sizes.idxmax()
    rows = df[run_id == best]
    return len(rows), s[run_id == best].iloc[0], rows["date"].min(), rows["date"].max()


def main():
    df = pd.read_csv(TRAIN)
    header("Overview")
    print(df.shape)
    print(df.dtypes)
    print(df.head())
    print(df.isna().sum())
    print(df.describe().T)

    header("1. Leaking column: columns missing from holdout_features.csv")
    if HOLDOUT.exists():
        hold = pd.read_csv(HOLDOUT)
        print("In train but not in holdout:", sorted(set(df.columns) - set(hold.columns)))
    if "pm25_measured_next_day" in df:
        print("corr with target:",
              df["pm25_measured_next_day"].corr(df[TARGET]))

    header("2. Unit mixup: median of each numeric column per station_zone")
    zone = df["station_zone"].astype(str).str.strip().str.lower()
    med = df.groupby(zone)[NUMERIC + [TARGET]].median()
    print(med)
    print("\nRatio to the overall median (a constant odd factor in one zone is a clue):")
    print((med / df[NUMERIC + [TARGET]].median()).round(2))

    header("3. Sentinel values: repeated values and extremes")
    for col in NUMERIC + [TARGET]:
        vc = df[col].value_counts()
        print(f"{col}: min={df[col].min()} max={df[col].max()} "
              f"top repeats={dict(vc[vc >= 3].head(4))}")

    header("4. Duplicate rows")
    print("full-row duplicates (extra copies):", df.duplicated().sum())
    print("record_id duplicates:", df.duplicated(subset="record_id").sum())

    header("5. Stuck sensor: longest run of identical values (sorted by date)")
    ds = df.copy()
    ds["date"] = pd.to_datetime(ds["date"])
    ds = ds.sort_values("date").reset_index(drop=True)
    for col in NUMERIC:
        n, val, start, end = longest_run(ds, col)
        print(f"{col}: run of {n} x {val}  from {start.date()} to {end.date()}")

    header("6. Target typos: target about 10x too big")
    ratio = df[TARGET] / df["pm25_today"]
    print(ratio.describe())
    print("target > 5 x pm25_today:", (ratio > 5).sum())
    print(df.loc[ratio > 5, ["record_id", "pm25_today", TARGET]].head(10))
    print("target > 3 x its 99th percentile of the rest, rows:",
          (df[TARGET] > 3 * df[TARGET].quantile(0.95)).sum())

    header("7. Spelling variants")
    for col in ["day_type", "station_zone"]:
        print(df[col].value_counts(dropna=False))
        print("after strip+lower:", df[col].astype(str).str.strip().str.lower().nunique(),
              "levels\n")


if __name__ == "__main__":
    main()
