"""Data loading, integrity checks and train/validation split."""
import hashlib

import pandas as pd
import yaml

from src.config import repo_path


def _file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    return _file_hash(path, "sha256")


def file_md5(path):
    return _file_hash(path, "md5")


def dvc_md5(dvc_file):
    """md5 recorded for the first output in a .dvc file."""
    with open(dvc_file) as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]


def load_train(params):
    """Load the original train.csv after checking it matches its .dvc file.

    Returns (df, info) where info holds the hashes to log with each run.
    """
    path = repo_path(params["data"]["train_path"])
    expected = dvc_md5(repo_path(params["data"]["dvc_file"]))
    actual = file_md5(path)
    if actual != expected:
        raise ValueError(
            f"{path} md5 {actual} does not match .dvc md5 {expected}. "
            "Run `dvc checkout` or re-track the data before training."
        )
    info = {"data_sha256": file_sha256(path), "data_md5_dvc": expected}
    df = pd.read_csv(path)

    if params["data"].get("use_clean", False):
        from src.clean import clean
        df = clean(df, params.get("cleaning", {}))

    return df, info


def time_split(df, date_col, test_size):
    """Chronological split: the most recent ~test_size of rows become validation.

    The cut is placed on a date boundary so no date appears on both sides.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    dates = pd.to_datetime(df[date_col])
    order = dates.sort_values(kind="stable")
    cutoff = order.iloc[int(len(order) * (1 - test_size))]
    train_mask = dates < cutoff
    if train_mask.all() or not train_mask.any():
        raise ValueError("Split produced an empty train or validation set.")
    return df[train_mask].copy(), df[~train_mask].copy()


def split_later(df, frac=0.7, date_col="date", id_col="record_id"):
    """Shared Stage 4 split of later.csv: first `frac` of rows by date vs the rest.

    v2 trains on the first part; v1 and v2 are both scored on the second part.
    Everyone must use this function so the rows line up exactly. Ties on the
    same date are broken by record_id, so the result is deterministic.
    """
    ordered = df.assign(_d=pd.to_datetime(df[date_col])).sort_values(
        ["_d", id_col], kind="stable").drop(columns="_d")
    cut = int(len(ordered) * frac)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
