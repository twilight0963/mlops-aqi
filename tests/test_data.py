import hashlib

import pandas as pd
import pytest

from src.config import load_params, repo_path
from src.data import dvc_md5, file_md5, file_sha256, time_split


def test_hashes_match_hashlib(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"a,b\n1,2\n")
    assert file_sha256(f) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert file_md5(f) == hashlib.md5(b"a,b\n1,2\n").hexdigest()


def test_dvc_md5_parsing(tmp_path):
    f = tmp_path / "train.csv.dvc"
    f.write_text("outs:\n- md5: abc123\n  size: 1\n  hash: md5\n  path: train.csv\n")
    assert dvc_md5(f) == "abc123"


def test_time_split_is_chronological_without_overlap():
    dates = pd.date_range("2026-01-01", periods=10).repeat(3)  # 3 rows per date
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "y": range(30)})
    df = df.sample(frac=1, random_state=0)  # shuffled input must not matter
    train, val = time_split(df, "date", 0.2)
    assert len(train) + len(val) == len(df)
    assert pd.to_datetime(train["date"]).max() < pd.to_datetime(val["date"]).min()
    assert set(train["date"]).isdisjoint(val["date"])


def test_time_split_rejects_bad_size():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"]})
    with pytest.raises(ValueError):
        time_split(df, "date", 1.5)


def test_params_never_use_leak_or_id_columns():
    p = load_params()
    feats = set(p["features"]["numeric"]) | set(p["features"]["categorical"])
    assert len(feats) == 8
    assert feats.isdisjoint(p["features"]["drop"])
    assert p["features"]["target"] not in feats
    assert "pm25_measured_next_day" in p["features"]["drop"]


def test_params_paths_exist():
    p = load_params()
    assert repo_path(p["data"]["dvc_file"]).exists()
