"""
COMP 3610 - Assignment 4: API Test Suite
Tests for the Taxi Tip Prediction FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# --- Valid sample input ---
VALID_INPUT = {
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


# --- Happy path tests ---

def test_root(client):
    """Root endpoint returns 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    """Health check reports healthy status and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_predict_valid(client):
    """Successful single prediction with valid input."""
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data
    assert isinstance(data["tip_amount"], float)
    assert data["tip_amount"] >= 0


def test_batch_prediction(client):
    """Successful batch prediction with multiple records."""
    records = [VALID_INPUT] * 3
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "processing_time_ms" in data
    for pred in data["predictions"]:
        assert "tip_amount" in pred
        assert "prediction_id" in pred


def test_model_info(client):
    """Model info endpoint returns expected metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features" in data
    assert "metrics" in data
    assert len(data["features"]) == 18


# --- Validation tests ---

def test_predict_missing_field(client):
    """Missing required field returns 422."""
    incomplete = {"pickup_hour": 14}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_type(client):
    """String where float expected returns 422."""
    invalid = VALID_INPUT.copy()
    invalid["trip_distance"] = "not a number"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_out_of_range_pickup_hour(client):
    """Pickup hour > 23 returns 422."""
    invalid = VALID_INPUT.copy()
    invalid["pickup_hour"] = 25
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_negative_distance(client):
    """Negative trip distance returns 422 (gt=0 constraint)."""
    invalid = VALID_INPUT.copy()
    invalid["trip_distance"] = -1.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_zero_distance(client):
    """Zero trip distance returns 422 (gt=0 means strictly positive)."""
    invalid = VALID_INPUT.copy()
    invalid["trip_distance"] = 0.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


# --- Edge case tests ---

def test_predict_extreme_fare(client):
    """Extreme but valid fare value still produces a prediction."""
    edge = VALID_INPUT.copy()
    edge["fare_amount"] = 499.99
    edge["trip_distance"] = 80.0
    edge["trip_duration_minutes"] = 120.0
    edge["trip_speed_mph"] = 40.0
    edge["log_trip_distance"] = 4.394
    edge["fare_per_mile"] = 6.25
    edge["fare_per_minute"] = 4.17
    response = client.post("/predict", json=edge)
    assert response.status_code == 200
    assert response.json()["tip_amount"] >= 0


def test_batch_max_limit(client):
    """Batch endpoint rejects > 100 records."""
    records = [VALID_INPUT] * 101
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 422
