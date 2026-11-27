
"""FastAPI application for next-day PM2.5 prediction."""

from contextlib import asynccontextmanager
import json
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_one

model = None


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"level": record.levelname, "event": record.getMessage(),
                           "logger": record.name}, ensure_ascii=False)


logger = logging.getLogger("mp221.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


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
        logger.info("model_loaded")
    except Exception as exc:
        logger.exception("model_load_failed")
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
        logger.info("prediction_succeeded")
        return {
            "predicted_pm25_next_day": round(prediction, 2),
            "unit": "µg/m³",
            "model": "mp221-pm25-model@champion",
        }
    except Exception as exc:
        logger.exception("prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc
