# Predictive Maintenance for Turbofan Engines

End-to-end machine learning system that predicts **Remaining Useful Life (RUL)** of aircraft turbofan engines using NASA C-MAPSS sensor data. Includes a trained Random Forest model, a FastAPI prediction backend, and a Streamlit monitoring dashboard fully running with Docker.

---

## Dataset

**NASA Turbofan Jet Engine (C-MAPSS) FD001**

| Property | Value |
|---|---|
| Training engines | 100 (run-to-failure) |
| Test engines | 100 (truncated histories) |
| Sensors per cycle | 21 sensor readings + 3 operational settings |
| Training rows | 20,631 cycles |
| Failure mode | Single (HPC degradation) |
| Operating condition | Single (sea level) |

Each engine starts healthy and degrades over time until failure. The goal is to predict how many cycles remain before failure at any given point.

## Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Trees | 300 (`n_estimators=300`) |
| Max depth | 10 |
| Engineered features | 91 (rolling mean/std over windows [5, 10] + lag features [1, 3]) |
| Target | RUL capped at 130 cycles |
| Test RMSE | **19.70 cycles** |
| Test MAE | 14.07 cycles |
| Test R² | 0.7752 |

### Feature Engineering Pipeline

```
Raw sensors (21) -> Drop low-variance (6 removed) → 15 sensors
  → Rolling mean/std (windows 5, 10): 60 features
  → Lag features (lags 1, 3): 30 features
  → + setting_1: 1 feature
  → Total: 91 features
```

## Project Structure

```
predictive-maintenance/
├── backend/
│   ├── main.py              # FastAPI prediction API
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit dashboard
│   ├── Dockerfile
│   └── requirements.txt
├── src/
│   ├── preprocess/basic.py  # Data loading & cleaning
│   ├── features/time_series.py  # Rolling stats & lag features
│   └── modeling/utils.py    # Scaling, splitting, training
├── notebooks/
│   ├── 1_eda_FD001.ipynb
│   ├── 2_preprocess.ipynb
│   ├── 3_feature_engineerin.ipynb
│   ├── 4_scaling_split.ipynb
│   └── 5_modeling.ipynb
├── data/
│   ├── raw/                 # NASA C-MAPSS text files 
│   ├── processed/           # Cleaned CSVs 
│   └── demo/                # Sample engine CSVs for testing
├── models/                  # Trained model & scaler 
├── figures/                 # EDA & model visualizations
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Conda (for local development)
- Docker & Docker Compose (for containerized deployment)

### Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/lohitharasakonda/predictive-maintenance.git
cd predictive-maintenance
```

**2. Create and activate the conda environment:**

```bash
conda create -n cmapss python=3.11 -y
conda activate cmapss
pip install -r requirements.txt
```

**3. Place model files** in the `models/` directory:

- `rf_model.pkl` — trained Random Forest model
- `scaler.pkl` — fitted StandardScaler

**4. Start the backend (Terminal 1):**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**5. Start the frontend (Terminal 2):**

```bash
cd frontend
streamlit run app.py
```

**6. Open** `http://localhost:8501` in your browser and upload a demo CSV from the sidebar.

### Run with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Health Check | http://localhost:8000/health |

## API Reference

### `POST /predict`

Upload a CSV file with engine sensor data. The API handles feature engineering, scaling, and prediction automatically.

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/demo/critical_engine.csv"
```

**Response:**

```json
{
  "status": "success",
  "num_rows": 200,
  "cycles": [4, 5, 6, ...],
  "predicted_rul": [130.0, 129.8, ...],
  "health_scores": [100.0, 99.8, ...],
  "latest": {
    "rul": 8.5,
    "health_score": 6.5,
    "status": "Critical"
  }
}
```

### `GET /health`

Returns API and model status.

## Demo Data

Three sample engine CSVs are included in `data/demo/` for testing:

| File | True RUL | Predicted RUL | Health Score | Status |
|---|---|---|---|---|
| `healthy_engine.csv` | 145 cycles | 119.1 | 91.6% | Healthy |
| `warning_engine.csv` | 21 cycles | 70.3 | 54.1% | Warning |
| `critical_engine.csv` | 7 cycles | 8.5 | 6.5% | Critical |

## Results

### Model Comparison

| Model | Val RMSE | Val MAE | Val R² |
|---|---|---|---|
| Linear Regression | 21.43 | 17.80 | 0.7560 |
| Ridge Regression (α=10) | 21.43 | 17.80 | 0.7560 |
| **Random Forest** | **18.82** | **13.93** | **0.8118** |

### Health Classification Thresholds

| Health Score | Status | Action |
|---|---|---|
| 60–100% | Healthy | No action needed |
| 30–60% | Warning | Schedule maintenance |
| 0–30% | Critical | Immediate maintenance |

## Screenshots

> _Screenshots of the Streamlit dashboard will be added here._

## Tech Stack

- **Machine Learning**: scikit-learn, pandas, NumPy
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **Containerization**: Docker
- **Data**: NASA C-MAPSS (Turbofan Engine Degradation Simulation)


