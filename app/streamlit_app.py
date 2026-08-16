"""
streamlit_app.py
================
Streamlit frontend for Student Performance Prediction.

Architecture:
  Streamlit -> FastAPI -> Saved ML Model

This application communicates exclusively via HTTP with the FastAPI backend.
"""

import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
PAGE_TITLE = "Student Performance Prediction System"

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional Enterprise CSS styling (Clean, non-AI aesthetic)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global layout & typography */
    body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Result Card */
    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
    }

    /* Score value */
    .score-value {
        font-size: 56px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    /* Performance badge */
    .badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 16px;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    .badge-excellent { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .badge-verygood  { background: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
    .badge-good      { background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
    .badge-average   { background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    .badge-poor      { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

    /* Metadata text */
    .meta-text {
        color: #64748b;
        font-size: 13px;
        margin-top: 8px;
    }

    /* Category band boxes */
    .band-box {
        border-radius: 6px;
        padding: 10px 6px;
        text-align: center;
        font-size: 13px;
    }

    /* Clean divider */
    hr {
        margin: 20px 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
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


def get_badge_html(perf: str) -> str:
    cls_map = {
        "EXCELLENT": "badge-excellent",
        "VERY GOOD": "badge-verygood",
        "GOOD": "badge-good",
        "AVERAGE": "badge-average",
        "POOR": "badge-poor",
    }
    css_class = cls_map.get(perf, "badge-average")
    return f'<div class="badge {css_class}">{perf}</div>'


# ---------------------------------------------------------------------------
# Sidebar — Model information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### System Status")

    backend_ok = api_health()
    if backend_ok:
        st.success("API Backend: Operational")
        info = api_model_info()
        if "error" not in info:
            st.markdown("---")
            st.markdown("### Active Model")
            st.write(f"**Architecture:** {info.get('model_name', 'N/A')}")
            st.write(f"**Version:** {info.get('model_version', 'N/A')}")
            st.write(f"**Type:** {info.get('model_type', 'regression')}")
            
            m = info.get("training_metrics", {})
            if m:
                st.markdown("---")
                st.markdown("### Evaluation Metrics")
                col1, col2 = st.columns(2)
                col1.metric("R-Squared", f"{m.get('r2', 0):.3f}")
                col2.metric("MAE", f"{m.get('mae', 0):.2f}")
                col1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
    else:
        st.error("API Backend: Offline")
        st.caption("Backend server is not responding at " + API_BASE_URL)

    st.markdown("---")
    st.markdown("### Project Metadata")
    st.write("**Course:** IT4V43 - Machine Learning Operations")
    st.write("**Tracking:** MLflow Tracking Server")
    st.write("**Pipeline:** DVC & Pandera Validation")


# ---------------------------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------------------------
st.title("Student Performance Prediction System")
st.markdown(
    "Academic performance evaluation system. Enter student metrics to estimate the expected "
    "final examination score using the verified regression pipeline."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
with st.form("student_profile_form"):
    st.subheader("Student Metrics Input")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider(
            "Study Hours (daily)",
            min_value=0.0,
            max_value=14.0,
            value=5.0,
            step=0.5,
            help="Average hours spent studying per day",
        )
        previous_exam_score = st.slider(
            "Previous Examination Score",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Score achieved in the preceding examination period",
        )
        sleep_hours = st.slider(
            "Sleep Duration (hours/night)",
            min_value=3.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            help="Average nightly sleep duration",
        )

    with col2:
        attendance_percentage = st.slider(
            "Attendance Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help="Cumulative classroom attendance percentage",
        )
        assignment_completion_percentage = st.slider(
            "Assignment Completion Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help="Percentage of submitted coursework and assignments",
        )
        extracurricular_hours = st.slider(
            "Extracurricular Activity (daily hours)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Time allocated to extracurricular and sports activities",
        )

    submitted = st.form_submit_button("Compute Prediction", use_container_width=True)


# ---------------------------------------------------------------------------
# Prediction Output
# ---------------------------------------------------------------------------
if submitted:
    if not backend_ok:
        st.error("Prediction request failed: API backend service is currently offline.")
    else:
        features = {
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "previous_exam_score": previous_exam_score,
            "assignment_completion_percentage": assignment_completion_percentage,
            "sleep_hours": sleep_hours,
            "extracurricular_hours": extracurricular_hours,
        }

        with st.spinner("Processing inference request via REST API..."):
            result = api_predict(features)

        if "error" in result:
            st.error(f"Inference error: {result['error']}")
        else:
            score = result["predicted_score"]
            perf = result["performance"]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.markdown("---")
            st.subheader("Prediction Analysis")

            model_name = info.get("model_name", "LinearRegression") if backend_ok else "Model"
            model_ver = info.get("model_version", "1.0.0") if backend_ok else "1.0.0"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="score-value">{score:.1f} / 100</div>
                    {get_badge_html(perf)}
                    <div class="meta-text">
                        Evaluated on {ts} | Model: {model_name} (v{model_ver})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Performance Classification Breakdown
            st.markdown("#### Performance Classification Scale")
            cats = [
                ("POOR", "0.0 - 39.9", "#fee2e2", "#991b1b"),
                ("AVERAGE", "40.0 - 59.9", "#ffedd5", "#9a3412"),
                ("GOOD", "60.0 - 74.9", "#fef9c3", "#854d0e"),
                ("VERY GOOD", "75.0 - 89.9", "#e0f2fe", "#075985"),
                ("EXCELLENT", "90.0 - 100.0", "#dcfce7", "#166534"),
            ]
            cols = st.columns(5)
            for i, (label, rng, bg, text_color) in enumerate(cats):
                with cols[i]:
                    active = (
                        (score < 40 and label == "POOR")
                        or (40 <= score < 60 and label == "AVERAGE")
                        or (60 <= score < 75 and label == "GOOD")
                        or (75 <= score < 90 and label == "VERY GOOD")
                        or (score >= 90 and label == "EXCELLENT")
                    )
                    border = f"2px solid {text_color}" if active else "1px solid #cbd5e1"
                    font_weight = "700" if active else "500"
                    st.markdown(
                        f"""
                        <div class="band-box" style="background:{bg}; color:{text_color}; border:{border}; font-weight:{font_weight};">
                            <div>{label}</div>
                            <div style="font-size:11px; opacity:0.85; margin-top:2px;">{rng}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Feature Importance Attribution
            st.markdown("---")
            st.subheader("Feature Contribution Summary (SHAP)")
            st.markdown(
                """
                Global feature importance computed via Shapley Additive Explanations:
                - **Previous Exam Score**: Primary baseline predictor for expected performance.
                - **Study Hours**: Strongest positive controllable factor influencing score.
                - **Attendance Percentage**: Significant correlation with concept retention.
                - **Assignment Completion**: Regular coursework submission provides positive reinforcement.
                - **Sleep Duration**: Adequate rest supports cognitive retention.
                - **Extracurricular Hours**: Moderate engagement is neutral; excessive hours show slight negative trade-off.
                """
            )

            # Input Parameters Table
            st.markdown("---")
            st.subheader("Submitted Input Parameters")
            input_df = pd.DataFrame(
                [
                    {"Feature": "Study Hours (daily)", "Value": f"{study_hours:.1f} hours"},
                    {"Feature": "Attendance Rate", "Value": f"{attendance_percentage:.0f}%"},
                    {"Feature": "Previous Exam Score", "Value": f"{previous_exam_score:.0f} / 100"},
                    {"Feature": "Assignment Completion Rate", "Value": f"{assignment_completion_percentage:.0f}%"},
                    {"Feature": "Sleep Duration", "Value": f"{sleep_hours:.1f} hours"},
                    {"Feature": "Extracurricular Activity", "Value": f"{extracurricular_hours:.1f} hours"},
                ]
            )
            st.dataframe(input_df, use_container_width=True, hide_index=True)
