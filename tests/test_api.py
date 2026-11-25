
from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_home_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AQI Prediction API is running"


def test_prediction_endpoint():
    payload = {
        "pm25_today": 120.0,
        "wind_kmph": 8.0,
        "humidity_pct": 65.0,
        "temp_c": 24.0,
        "traffic_index": 7.0,
        "stubble_fires": 2.0,
        "day_type": "weekday",
        "station_zone": "Central",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["predicted_pm25_next_day"], float)
    assert result["unit"] == "µg/m³"


def test_invalid_humidity():
    payload = {
        "pm25_today": 120.0,
        "wind_kmph": 8.0,
        "humidity_pct": 150.0,
        "temp_c": 24.0,
        "traffic_index": 7.0,
        "stubble_fires": 2.0,
        "day_type": "weekday",
        "station_zone": "Central",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_day_type():
    payload = {
        "pm25_today": 120.0,
        "wind_kmph": 8.0,
        "humidity_pct": 65.0,
        "temp_c": 24.0,
        "traffic_index": 7.0,
        "stubble_fires": 2.0,
        "day_type": "holiday",
        "station_zone": "Central",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
