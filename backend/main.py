"""
FastAPI backend for Engine RUL Prediction.
Accepts raw sensor CSV uploads, applies feature engineering, and returns predictions.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import os

app = FastAPI(title="Engine RUL Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# --- Constants matching training pipeline ---
SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

LOW_VARIANCE_COLS = [
    "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19",
    "setting_2", "setting_3",
]

# Feature column order must match what the scaler was fitted on
FEATURE_COLUMNS = ["setting_1"]
for window in [5, 10]:
    for sensor in SENSORS:
        FEATURE_COLUMNS.append(f"{sensor}_mean_{window}")
        FEATURE_COLUMNS.append(f"{sensor}_std_{window}")
for lag in [1, 3]:
    for sensor in SENSORS:
        FEATURE_COLUMNS.append(f"{sensor}_lag_{lag}")

MAX_RUL = 130  # RUL cap used during training


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering pipeline used during training:
    rolling statistics (mean, std) over windows [5, 10] and lag features [1, 3].
    """
    df = df.copy()
    df = df.sort_values(["engine_number", "cycle"]).reset_index(drop=True)

    # Drop low-variance columns if present (raw data)
    for col in LOW_VARIANCE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    available_sensors = [s for s in SENSORS if s in df.columns]
    if not available_sensors:
        raise ValueError("No recognized sensor columns found in uploaded data.")

    # Rolling statistics
    for window in [5, 10]:
        for sensor in available_sensors:
            df[f"{sensor}_mean_{window}"] = df.groupby("engine_number")[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{sensor}_std_{window}"] = df.groupby("engine_number")[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

    # Lag features
    for lag_val in [1, 3]:
        for sensor in available_sensors:
            df[f"{sensor}_lag_{lag_val}"] = df.groupby("engine_number")[sensor].shift(lag_val)

    # Drop original sensor columns (model was trained without them)
    df = df.drop(columns=available_sensors)

    # Drop rows with NaN from lags/rolling std
    df = df.dropna().reset_index(drop=True)

    return df


def compute_health(rul: float) -> dict:
    """Convert predicted RUL to health score and status."""
    health_score = min(100.0, max(0.0, (rul / MAX_RUL) * 100))
    if health_score >= 60:
        status = "Healthy"
    elif health_score >= 30:
        status = "Warning"
    else:
        status = "Critical"
    return {"health_score": round(health_score, 1), "status": status}


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "RandomForest", "features": len(FEATURE_COLUMNS)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode()))

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        # Detect whether data needs feature engineering
        has_raw_sensors = any(s in df.columns for s in SENSORS)

        if has_raw_sensors:
            # Raw sensor data — run full pipeline
            if "engine_number" not in df.columns:
                df["engine_number"] = 1
            if "cycle" not in df.columns:
                df["cycle"] = range(1, len(df) + 1)

            df_feat = engineer_features(df)
        else:
            # Already feature-engineered data
            df_feat = df.copy()

        # Extract cycles for response
        cycles = df_feat["cycle"].tolist() if "cycle" in df_feat.columns else list(range(1, len(df_feat) + 1))

        # Validate feature columns
        missing = set(FEATURE_COLUMNS) - set(df_feat.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing {len(missing)} required feature columns. First 5: {sorted(missing)[:5]}"
            )

        X = df_feat[FEATURE_COLUMNS]

        # Scale and predict (use DataFrame to preserve feature names)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLUMNS)
        predictions = model.predict(X_scaled)
        predictions = np.clip(predictions, 0, MAX_RUL)

        # Compute health scores
        health_scores = np.clip(predictions / MAX_RUL * 100, 0, 100)

        # Latest prediction (most relevant — current engine state)
        latest = compute_health(float(predictions[-1]))
        latest["rul"] = round(float(predictions[-1]), 1)

        return JSONResponse({
            "status": "success",
            "num_rows": len(predictions),
            "cycles": cycles,
            "predicted_rul": [round(float(p), 1) for p in predictions],
            "health_scores": [round(float(h), 1) for h in health_scores],
            "latest": latest,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
