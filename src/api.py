
"""FastAPI application for next-day PM2.5 prediction."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_one

model = None


class PredictionInput(BaseModel):
    pm25_today: float = Field(ge=0, le=2000)
    wind_kmph: float = Field(ge=0, le=500)
    humidity_pct: float = Field(ge=0, le=100)
    temp_c: float = Field(ge=-50, le=70)
    traffic_index: float = Field(ge=0, le=10)
    stubble_fires: float = Field(ge=0)
    day_type: str = Field(pattern="^(weekday|weekend|festival)$")
    station_zone: str = Field(
        pattern="^(Industrial|Central|Residential)$"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model("models/model.joblib")
        # model = load_model("champion")
        print("Champion model loaded successfully.")
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        raise
    yield
    model = None


app = FastAPI(
    title="AQI Prediction API",
    description="Predict next-day PM2.5 using the registered ML model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "AQI Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available",
        )

    try:
        prediction = predict_one(
            model,
            request.model_dump(),
        )
        return {
            "predicted_pm25_next_day": round(prediction, 2),
            "unit": "µg/m³",
            "model": "mp221-pm25-model@champion",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
