
import pandas as pd
from src.drift import calculate_psi, mae_by_group


def test_psi_same_data():
    data = [10, 20, 30, 40, 50]
    result = calculate_psi(data, data)
    assert abs(result) < 1e-6


def test_mae_by_group():
    df = pd.DataFrame({
        "zone": ["A", "A", "B", "B"],
        "actual": [10, 20, 30, 40],
        "predicted": [12, 18, 33, 37]
    })

    result = mae_by_group(df, "actual", "predicted", "zone")

    assert result.loc[result["zone"] == "A", "mae"].iloc[0] == 2
    assert result.loc[result["zone"] == "B", "mae"].iloc[0] == 3


def test_psi_includes_values_outside_reference_range():
    from src.drift import calculate_psi
    assert calculate_psi([1, 2, 3, 4], [0, 0, 0, 100], bins=2) > 0


def test_mae_defaults_to_station_zone():
    df = pd.DataFrame({"station_zone": ["Central"], "actual": [12], "predicted": [10]})
    assert mae_by_group(df, "actual", "predicted").iloc[0]["mae"] == 2
