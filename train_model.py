import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

print("Loading data...")
df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
zl = pd.read_csv("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv")
print(f"Loaded {len(df):,}")

d = df[df["payment_type"] == 1].copy()
d = d[(d["trip_distance"] > 0) & (d["trip_distance"] <= 100)]
d = d[d["fare_amount"] > 0]
d = d[(d["passenger_count"] >= 1) & (d["passenger_count"] <= 6)]
d["tpep_pickup_datetime"] = pd.to_datetime(d["tpep_pickup_datetime"])
d["tpep_dropoff_datetime"] = pd.to_datetime(d["tpep_dropoff_datetime"])
d["trip_duration_seconds"] = (d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]).dt.total_seconds()
d = d[(d["trip_duration_seconds"] >= 60) & (d["trip_duration_seconds"] <= 10800)]
d = d[(d["tip_amount"] >= 0) & (d["tip_amount"] <= 100)].reset_index(drop=True)
print(f"Clean: {len(d):,}")

d = d.sample(n=200000, random_state=42).reset_index(drop=True)
print(f"Sample: {len(d):,}")

d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour
d["pickup_day_of_week"] = d["tpep_pickup_datetime"].dt.dayofweek
d["is_weekend"] = d["pickup_day_of_week"].isin([5, 6]).astype(int)
d["trip_duration_minutes"] = d["trip_duration_seconds"] / 60.0
d["trip_speed_mph"] = np.where(d["trip_duration_minutes"] > 0, d["trip_distance"] / (d["trip_duration_minutes"] / 60.0), 0)
d["trip_speed_mph"] = d["trip_speed_mph"].clip(upper=100)
d["log_trip_distance"] = np.log1p(d["trip_distance"])
d["fare_per_mile"] = np.where(d["trip_distance"] > 0, d["fare_amount"] / d["trip_distance"], 0)
d["fare_per_mile"] = d["fare_per_mile"].clip(upper=100)
d["fare_per_minute"] = np.where(d["trip_duration_minutes"] > 0, d["fare_amount"] / d["trip_duration_minutes"], 0)
d["fare_per_minute"] = d["fare_per_minute"].clip(upper=50)

zd = zl.set_index("LocationID")["Borough"].to_dict()
d["pickup_borough"] = d["PULocationID"].map(zd).fillna("Unknown")
d["dropoff_borough"] = d["DOLocationID"].map(zd).fillna("Unknown")
le1 = LabelEncoder()
le2 = LabelEncoder()
d["pickup_borough_encoded"] = le1.fit_transform(d["pickup_borough"])
d["dropoff_borough_encoded"] = le2.fit_transform(d["dropoff_borough"])

FEATURES = [
    "pickup_hour", "pickup_day_of_week", "is_weekend", "trip_distance",
    "trip_duration_minutes", "trip_speed_mph", "log_trip_distance",
    "fare_amount", "fare_per_mile", "fare_per_minute", "passenger_count",
    "pickup_borough_encoded", "dropoff_borough_encoded", "tolls_amount",
    "extra", "mta_tax", "congestion_surcharge", "Airport_fee"
]

for c in FEATURES:
    if d[c].isnull().any():
        d[c] = d[c].fillna(d[c].median())

X = d[FEATURES].values
y = d["tip_amount"].values

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Training Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_split=10,
    min_samples_leaf=5, n_jobs=-1, random_state=42
)
rf.fit(X_train_s, y_train)
print("Training complete.")

preds = rf.predict(X_test_s)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(FEATURES, "models/feature_names.joblib")
print("Saved model artifacts to models/")

model_size = os.path.getsize("models/model.joblib") / 1024 / 1024
print(f"Model size: {model_size:.1f} MB")
