"""
streamlit_app.py
================
Streamlit frontend for Student Performance Prediction.

Architecture:
  Streamlit → FastAPI → Saved ML Model

This app does NOT load the ML model directly.
All predictions are obtained by calling the FastAPI backend via HTTP.
"""

import os
import time
from datetime import datetime

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Reads from environment variable so Docker Compose can set it to
# http://backend:8000 while local dev defaults to http://localhost:8000
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
PAGE_TITLE = "Student Performance Predictor"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* Card */
    .result-card {
        background: rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin: 16px 0;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
    }

    /* Score display */
    .score-value {
        font-size: 72px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #f6d365, #fda085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Performance badge */
    .badge-excellent { color: #00e676; font-weight: 700; font-size: 24px; text-align: center; }
    .badge-verygood  { color: #40c4ff; font-weight: 700; font-size: 24px; text-align: center; }
    .badge-good      { color: #b2ff59; font-weight: 700; font-size: 24px; text-align: center; }
    .badge-average   { color: #ffab40; font-weight: 700; font-size: 24px; text-align: center; }
    .badge-poor      { color: #ff5252; font-weight: 700; font-size: 24px; text-align: center; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.4);
    }

    /* Headers */
    h1, h2, h3 { color: #fff !important; }

    /* Footer metadata */
    .meta-text { color: #aaa; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: API calls
# ---------------------------------------------------------------------------
def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "healthy"
    except Exception:
        return False


def api_model_info() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_predict(features: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE_URL}/predict", json=features, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Badge helper
# ---------------------------------------------------------------------------
def performance_badge(perf: str) -> str:
    cls = {
        "EXCELLENT": "badge-excellent",
        "VERY GOOD": "badge-verygood",
        "GOOD": "badge-good",
        "AVERAGE": "badge-average",
        "POOR": "badge-poor",
    }.get(perf, "badge-average")
    return f'<div class="{cls}">{perf}</div>'


# ---------------------------------------------------------------------------
# Sidebar — model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Model Information")

    backend_ok = api_health()
    if backend_ok:
        st.success("✅ API Backend: Online")
        info = api_model_info()
        if "error" not in info:
            st.markdown(f"**Model:** `{info.get('model_name', '—')}`")
            st.markdown(f"**Version:** `{info.get('model_version', '—')}`")
            st.markdown(f"**Type:** `{info.get('model_type', '—')}`")
            m = info.get("training_metrics", {})
            if m:
                st.markdown("---")
                st.markdown("**Training Metrics**")
                col1, col2 = st.columns(2)
                col1.metric("R²", f"{m.get('r2', 0):.3f}")
                col2.metric("MAE", f"{m.get('mae', 0):.2f}")
                col1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
    else:
        st.error("❌ API Backend: Offline")
        st.warning("Start FastAPI:\n```\nuvicorn api.main:app\n```")

    st.markdown("---")
    st.markdown("**IT4V43 — MLOps Project**")
    st.markdown("*Student Performance Prediction*")
    st.markdown("[GitHub Actions CI/CD]() | [MLflow Tracking](http://localhost:5000)")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("# 🎓 Student Performance Predictor")
st.markdown(
    "Enter a student's academic profile below to predict their final exam score. "
    "Predictions are served by a **FastAPI** backend connected to a trained **Scikit-learn** model."
)

st.markdown("---")

# ── Input form ──────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.markdown("### 📋 Student Profile")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider(
            "📚 Study Hours (per day)", min_value=0.0, max_value=14.0, value=5.0, step=0.5,
            help="Average daily hours spent studying"
        )
        previous_exam_score = st.slider(
            "📝 Previous Exam Score", min_value=0.0, max_value=100.0, value=65.0, step=1.0,
            help="Score on the most recent exam"
        )
        sleep_hours = st.slider(
            "😴 Sleep Hours (per night)", min_value=3.0, max_value=12.0, value=7.0, step=0.5,
            help="Average nightly sleep duration"
        )

    with col2:
        attendance_percentage = st.slider(
            "🏫 Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0,
            help="Percentage of classes attended"
        )
        assignment_completion_percentage = st.slider(
            "✅ Assignment Completion (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0,
            help="Percentage of assignments completed"
        )
        extracurricular_hours = st.slider(
            "⚽ Extracurricular Hours (per day)", min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            help="Daily hours in extracurricular activities"
        )

    submitted = st.form_submit_button("🚀 Predict Final Score", use_container_width=True)


# ── Prediction result ───────────────────────────────────────────────────────
if submitted:
    if not backend_ok:
        st.error("Cannot make predictions — API backend is offline. Start FastAPI first.")
    else:
        features = {
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "previous_exam_score": previous_exam_score,
            "assignment_completion_percentage": assignment_completion_percentage,
            "sleep_hours": sleep_hours,
            "extracurricular_hours": extracurricular_hours,
        }

        with st.spinner("Calling FastAPI backend ..."):
            result = api_predict(features)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            score = result["predicted_score"]
            perf = result["performance"]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="score-value">{score}</div>
                    {performance_badge(perf)}
                    <br/>
                    <p class="meta-text" style="text-align:center">
                        Predicted at {ts}<br/>
                        Model: {info.get('model_name', '—')} v{info.get('model_version', '—')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Performance guide
            st.markdown("---")
            st.markdown("### 🏆 Performance Categories")
            cats = [
                ("🔴 POOR", "0–39", "#ff5252"),
                ("🟠 AVERAGE", "40–59", "#ffab40"),
                ("🟡 GOOD", "60–74", "#b2ff59"),
                ("🔵 VERY GOOD", "75–89", "#40c4ff"),
                ("🟢 EXCELLENT", "90–100", "#00e676"),
            ]
            cols = st.columns(5)
            for i, (label, rng, color) in enumerate(cats):
                with cols[i]:
                    active = (
                        (score < 40 and "POOR" in label)
                        or (40 <= score < 60 and "AVERAGE" in label)
                        or (60 <= score < 75 and "GOOD" in label and "VERY" not in label)
                        or (75 <= score < 90 and "VERY GOOD" in label)
                        or (score >= 90 and "EXCELLENT" in label)
                    )
                    border = f"3px solid {color}" if active else "1px solid #444"
                    st.markdown(
                        f'<div style="border:{border};border-radius:8px;padding:8px;text-align:center;">'
                        f'<b>{label}</b><br/><small>{rng}</small></div>',
                        unsafe_allow_html=True,
                    )

            # Feature summary
            st.markdown("---")
            st.markdown("### 🔍 Feature Explanation (SHAP Ranking)")
            st.info(
                "Based on SHAP analysis of the trained model, the most influential features are:\n\n"
                "1. **previous_exam_score** — strongest predictor (past performance predicts future)\n"
                "2. **study_hours** — more study time → higher score\n"
                "3. **attendance_percentage** — consistent attendance boosts learning\n"
                "4. **assignment_completion_percentage** — completing work reinforces knowledge\n"
                "5. **sleep_hours** — adequate sleep improves retention\n"
                "6. **extracurricular_hours** — small negative effect when excessive\n\n"
                "*Individual SHAP plots are available in the `models/` directory after training.*"
            )

            # Input summary table
            st.markdown("---")
            st.markdown("### 📌 Input Summary")
            import pandas as pd
            input_df = pd.DataFrame([features]).T.rename(columns={0: "Value"})
            st.dataframe(input_df, use_container_width=True)
