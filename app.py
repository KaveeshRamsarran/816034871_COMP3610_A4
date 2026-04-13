"""
COMP 3610 - Assignment 4: FastAPI Prediction Service
Serves tip_amount predictions for NYC Yellow Taxi trips using a trained Random Forest model.
"""

import os
import uuid
import time
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
MODEL_VERSION = "1.0.0"
MODEL_NAME = "taxi-tip-regressor"

# Feature order must match training
FEATURE_COLUMNS = [
    "pickup_hour", "pickup_day_of_week", "is_weekend",
    "trip_distance", "trip_duration_minutes", "trip_speed_mph", "log_trip_distance",
    "fare_amount", "fare_per_mile", "fare_per_minute",
    "passenger_count",
    "pickup_borough_encoded", "dropoff_borough_encoded",
    "tolls_amount", "extra", "mta_tax", "congestion_surcharge", "Airport_fee",
]

# Training metrics from Assignment 2 (Random Forest Regressor on test set)
TRAINING_METRICS = {"MAE": 1.1825, "RMSE": 2.2700, "R2": 0.6407}

# --- Global State ---
ml_model = None
scaler = None
start_time = None


# --- Pydantic Models ---
class TripInput(BaseModel):
    """Input schema for a single taxi trip prediction."""
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0 or 1)")
    trip_distance: float = Field(..., gt=0, le=100, description="Trip distance in miles")
    trip_duration_minutes: float = Field(..., gt=0, le=1440, description="Trip duration in minutes")
    trip_speed_mph: float = Field(..., ge=0, le=200, description="Average speed in mph")
    log_trip_distance: float = Field(..., ge=0, description="Log-transformed trip distance")
    fare_amount: float = Field(..., ge=0, le=500, description="Base fare amount in dollars")
    fare_per_mile: float = Field(..., ge=0, le=100, description="Fare divided by distance")
    fare_per_minute: float = Field(..., ge=0, le=50, description="Fare divided by duration")
    passenger_count: int = Field(..., ge=1, le=9, description="Number of passengers")
    pickup_borough_encoded: int = Field(..., ge=0, le=6, description="Encoded pickup borough")
    dropoff_borough_encoded: int = Field(..., ge=0, le=6, description="Encoded dropoff borough")
    tolls_amount: float = Field(..., ge=0, le=200, description="Toll charges in dollars")
    extra: float = Field(..., ge=-5, le=20, description="Extra charges")
    mta_tax: float = Field(..., ge=0, le=5, description="MTA tax")
    congestion_surcharge: float = Field(..., ge=0, le=5, description="Congestion surcharge")
    Airport_fee: float = Field(..., ge=0, le=5, description="Airport fee")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pickup_hour": 14,
                    "pickup_day_of_week": 2,
                    "is_weekend": 0,
                    "trip_distance": 3.5,
                    "trip_duration_minutes": 15.0,
                    "trip_speed_mph": 14.0,
                    "log_trip_distance": 1.504,
                    "fare_amount": 18.0,
                    "fare_per_mile": 5.14,
                    "fare_per_minute": 1.2,
                    "passenger_count": 1,
                    "pickup_borough_encoded": 3,
                    "dropoff_borough_encoded": 3,
                    "tolls_amount": 0.0,
                    "extra": 1.0,
                    "mta_tax": 0.5,
                    "congestion_surcharge": 2.5,
                    "Airport_fee": 0.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""
    tip_amount: float
    prediction_id: str
    model_version: str


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    records: List[TripInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


# --- Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and scaler once at startup."""
    global ml_model, scaler, start_time
    ml_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    start_time = time.time()
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Scaler loaded from {SCALER_PATH}")
    yield
    print("Shutting down...")


# --- App ---
app = FastAPI(
    title="Taxi Tip Prediction API",
    description="Predicts tip amounts for NYC Yellow Taxi trips using a Random Forest model trained on January 2024 data.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# --- Helper ---
def _predict_single(input_data: TripInput) -> PredictionResponse:
    """Run prediction for a single trip record."""
    features = np.array([[
        input_data.pickup_hour,
        input_data.pickup_day_of_week,
        input_data.is_weekend,
        input_data.trip_distance,
        input_data.trip_duration_minutes,
        input_data.trip_speed_mph,
        input_data.log_trip_distance,
        input_data.fare_amount,
        input_data.fare_per_mile,
        input_data.fare_per_minute,
        input_data.passenger_count,
        input_data.pickup_borough_encoded,
        input_data.dropoff_borough_encoded,
        input_data.tolls_amount,
        input_data.extra,
        input_data.mta_tax,
        input_data.congestion_surcharge,
        input_data.Airport_fee,
    ]])
    features_scaled = scaler.transform(features)
    prediction = ml_model.predict(features_scaled)[0]
    # Ensure tip is non-negative
    prediction = max(0.0, float(prediction))
    return PredictionResponse(
        tip_amount=round(prediction, 2),
        prediction_id=str(uuid.uuid4()),
        model_version=MODEL_VERSION,
    )


# --- Endpoints ---
@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Taxi Tip Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripInput):
    """Predict tip amount for a single taxi trip."""
    return _predict_single(input_data)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    """Predict tip amounts for a batch of taxi trips (max 100)."""
    start = time.time()
    predictions = [_predict_single(record) for record in batch.records]
    elapsed = (time.time() - start) * 1000
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed, 2),
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0,
    }


@app.get("/model/info")
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "features": FEATURE_COLUMNS,
        "metrics": TRAINING_METRICS,
        "trained_date": "2024-01-31",
        "dataset": "NYC Yellow Taxi Trip Records - January 2024",
        "algorithm": "RandomForestRegressor",
    }


# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unexpected errors and return a structured JSON response."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )
