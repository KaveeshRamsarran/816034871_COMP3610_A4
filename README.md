# COMP 3610 - Assignment 4: MLOps & Model Deployment

**Student:** Kaveesh Ramsarran (816034871)  
**Course:** COMP 3610 - Big Data Analytics  
**Semester:** II, 2025-2026

## Overview

This project deploys a taxi tip prediction model as a containerized REST API. It covers:

1. **MLflow Experiment Tracking** - Logging, comparing, and registering models
2. **FastAPI Prediction Service** - REST API with input validation and error handling
3. **Docker Containerization** - Dockerfile and Docker Compose for reproducible deployment

The model predicts `tip_amount` for NYC Yellow Taxi trips using a Random Forest Regressor trained on January 2024 data (from Assignment 2).

## Prerequisites

- Python 3.10+
- Docker Desktop installed and running
- pip package manager

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

Open `assignment4.ipynb` and run all cells. This will:
- Download NYC Taxi data and train models
- Log experiments to MLflow
- Save the best model to `models/`

### 3. Start MLflow UI (optional)

```bash
mlflow ui --port 5000
```

### 4. Run the API Locally

```bash
uvicorn app:app --reload --port 8000
```

Visit http://localhost:8000/docs for the Swagger UI.

### 5. Run Tests

```bash
pytest test_app.py -v
```

### 6. Docker Deployment

```bash
# Build and start all services
docker compose up --build -d

# Make a test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pickup_hour":14,"pickup_day_of_week":2,"is_weekend":0,"trip_distance":3.5,"trip_duration_minutes":15.0,"trip_speed_mph":14.0,"log_trip_distance":1.504,"fare_amount":18.0,"fare_per_mile":5.14,"fare_per_minute":1.2,"passenger_count":1,"pickup_borough_encoded":3,"dropoff_borough_encoded":3,"tolls_amount":0.0,"extra":1.0,"mta_tax":0.5,"congestion_surcharge":2.5,"Airport_fee":0.0}'

# Check health
curl http://localhost:8000/health

# Stop services
docker compose down
```

## Project Structure

```
816034871_COMP3610_A4/
├── assignment4.ipynb       ← Jupyter notebook with full documentation
├── app.py                  ← FastAPI application
├── test_app.py             ← pytest test suite
├── Dockerfile              ← Container definition
├── docker-compose.yml      ← Multi-service orchestration (API + MLflow)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Git exclusions
├── .dockerignore           ← Docker build exclusions
└── models/                 ← Saved model artifacts (gitignored)
    ├── model.joblib
    ├── scaler.joblib
    └── feature_names.joblib
```

## API Endpoints

| Method | Endpoint         | Description                          |
|--------|------------------|--------------------------------------|
| GET    | `/`              | Root - API status message            |
| POST   | `/predict`       | Single trip tip prediction           |
| POST   | `/predict/batch` | Batch predictions (up to 100 trips)  |
| GET    | `/health`        | Health check with model status       |
| GET    | `/model/info`    | Model metadata and training metrics  |

## Model Details

- **Algorithm:** Random Forest Regressor
- **Features:** 18 engineered features from NYC taxi trip data
- **Test Metrics:** MAE=$1.18, RMSE=$2.27, R²=0.64
- **Dataset:** NYC Yellow Taxi Trip Records (January 2024)
