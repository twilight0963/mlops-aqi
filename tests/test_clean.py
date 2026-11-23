import pandas as pd

from src.clean import clean


def sample_data():
    return pd.DataFrame({
        "record_id": [1, 2, 2],
        "day_type": [" WEEKDAY ", "Weekend", "Weekend"],
        "station_zone": ["industrial", " CENTRAL ", " CENTRAL "],
        "pm25_today": [50, 60, 60],
        "pm25_measured_next_day": [55, 65, 65],
        "pm25_next_day": [55, 65, 65],
    })


def test_clean_removes_duplicates():
    df = sample_data()
    result = clean(df)

    assert len(result) == 2
    assert result.duplicated().sum() == 0


def test_clean_normalizes_categories():
    result = clean(sample_data())

    assert result["day_type"].tolist() == ["weekday", "weekend"]
    assert result["station_zone"].tolist() == ["Industrial", "Central"]


def test_clean_drops_leakage_column():
    result = clean(sample_data())

    assert "pm25_measured_next_day" not in result.columns
    assert "pm25_next_day" in result.columns


def test_clean_does_not_modify_original_dataframe():
    df = sample_data()
    original = df.copy(deep=True)

    clean(df)

    pd.testing.assert_frame_equal(df, original)


def test_clean_corrects_central_unit_mixup():
    df = pd.DataFrame({
        "station_zone": ["Central", "Industrial"],
        "pm25_today": [166500.0, 156.3],
    })

    result = clean(df, {
        "unit_mixup": {
            "group_column": "station_zone",
            "group_value": "central",
            "column": "pm25_today",
            "wrong_value_min": 1000,
            "factor": 1000,
        }
    })

    assert result["pm25_today"].tolist() == [166.5, 156.3]


def test_clean_replaces_wind_sentinel():
    df = pd.DataFrame({"wind_kmph": [12.5, 9999.0]})

    result = clean(df, {"sentinels": {"wind_kmph": [9999]}})

    assert result["wind_kmph"].iloc[0] == 12.5
    assert pd.isna(result["wind_kmph"].iloc[1])
