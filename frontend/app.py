"""
Streamlit frontend for Engine RUL Prediction.
Upload engine sensor data to view predicted RUL, health score, and sensor trends.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
DEMO_DIR = os.environ.get("DEMO_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "demo"))

st.set_page_config(page_title="Engine Health Monitor", layout="wide")
st.title("Aircraft Engine Predictive Maintenance")
st.markdown("Upload engine sensor data to predict **Remaining Useful Life (RUL)** and assess engine health.")

# --- Sidebar: Demo Downloads ---
with st.sidebar:
    st.header("Demo Data")
    st.markdown("Download sample engine CSVs to try out predictions:")
    demos = [
        ("Healthy Engine (RUL ~145)", "healthy_engine.csv"),
        ("Warning Engine (RUL ~50)", "warning_engine.csv"),
        ("Critical Engine (RUL ~7)", "critical_engine.csv"),
    ]
    for label, fname in demos:
        fpath = os.path.join(DEMO_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                st.download_button(label, f, file_name=fname, mime="text/csv")

    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Algorithm: Random Forest (300 trees)")
    st.markdown("- Test RMSE: 19.70 cycles")
    st.markdown("- Features: 91 engineered features")
    st.markdown("- Dataset: NASA C-MAPSS FD001")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Engine Sensor CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{len(df)} rows x {len(df.columns)} columns")

    if st.button("Predict RUL", type="primary", use_container_width=True):
        with st.spinner("Running prediction pipeline..."):
            uploaded_file.seek(0)
            files = {"file": ("data.csv", uploaded_file, "text/csv")}
            try:
                resp = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend API. Ensure the FastAPI server is running on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        if result.get("status") != "success":
            st.error(f"API error: {result}")
            st.stop()

        latest = result["latest"]
        rul = latest["rul"]
        health = latest["health_score"]
        status = latest["status"]

        # --- Status Banner ---
        if status == "Healthy":
            st.success(f"Engine Status: **{status}** — No immediate maintenance required.")
        elif status == "Warning":
            st.warning(f"Engine Status: **{status}** — Schedule maintenance soon.")
        else:
            st.error(f"Engine Status: **{status}** — Immediate maintenance recommended!")

        # --- Key Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted RUL", f"{rul} cycles")
        col2.metric("Health Score", f"{health}%")
        col3.metric("Rows Analyzed", result["num_rows"])

        # --- Health Gauge ---
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            number={"suffix": "%"},
            title={"text": "Engine Health Score"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#2c3e50"},
                "steps": [
                    {"range": [0, 30], "color": "#e74c3c"},
                    {"range": [30, 60], "color": "#f39c12"},
                    {"range": [60, 100], "color": "#2ecc71"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": health,
                },
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- RUL Trend ---
        result_df = pd.DataFrame({
            "Cycle": result["cycles"],
            "Predicted RUL": result["predicted_rul"],
            "Health Score": result["health_scores"],
        })

        fig_rul = px.line(
            result_df, x="Cycle", y="Predicted RUL",
            title="Predicted RUL Over Engine Cycles",
            labels={"Predicted RUL": "RUL (cycles)"},
        )
        fig_rul.update_traces(line_color="#3498db")
        fig_rul.update_layout(height=400)
        st.plotly_chart(fig_rul, use_container_width=True)

        # --- Health Score Trend ---
        fig_health = go.Figure()
        fig_health.add_trace(go.Scatter(
            x=result_df["Cycle"], y=result_df["Health Score"],
            fill="tozeroy", mode="lines",
            line={"color": "#2ecc71"},
            fillcolor="rgba(46, 204, 113, 0.3)",
        ))
        # Add threshold lines
        fig_health.add_hline(y=60, line_dash="dash", line_color="#f39c12",
                             annotation_text="Warning Threshold (60%)")
        fig_health.add_hline(y=30, line_dash="dash", line_color="#e74c3c",
                             annotation_text="Critical Threshold (30%)")
        fig_health.update_layout(
            title="Health Score Over Engine Cycles",
            xaxis_title="Cycle", yaxis_title="Health Score (%)",
            yaxis_range=[0, 105], height=400,
        )
        st.plotly_chart(fig_health, use_container_width=True)

        # --- Sensor Trends (from uploaded raw data) ---
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        if sensor_cols and "cycle" in df.columns:
            st.subheader("Sensor Trends")
            default_sensors = sensor_cols[:4] if len(sensor_cols) >= 4 else sensor_cols
            selected = st.multiselect("Select sensors to visualize", sensor_cols, default=default_sensors)
            if selected:
                fig_sensors = px.line(
                    df, x="cycle", y=selected,
                    title="Raw Sensor Readings Over Time",
                    labels={"value": "Sensor Reading", "variable": "Sensor"},
                )
                fig_sensors.update_layout(height=400)
                st.plotly_chart(fig_sensors, use_container_width=True)

        # --- Prediction Table ---
        with st.expander("Full Prediction Table"):
            display_df = result_df.copy()
            display_df["Status"] = display_df["Health Score"].apply(
                lambda h: "Healthy" if h >= 60 else ("Warning" if h >= 30 else "Critical")
            )
            st.dataframe(display_df, use_container_width=True)
else:
    st.info("Upload a CSV file with engine sensor readings to get started. Download demo data from the sidebar.")
