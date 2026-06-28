"""
scoring.py

Batch Scoring page for the
Enterprise Customer Risk Prediction System.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# ==========================================================
# Settings
# ==========================================================

from src.config.settings import (
    BATCH_MONITORING_PATH,
    MODEL_PATH,
    PREDICTION_LOG_PATH,
    REFERENCE_PROFILE_PATH,
)

# ==========================================================
# Monitoring
# ==========================================================

from src.monitoring.drift_detection import (
    load_profile,
    save_profile,
    summarize_batch,
)

from src.monitoring.model_performance import (
    append_prediction_log,
)

# ==========================================================
# Prediction
# ==========================================================

from src.models.predict import predict_churn
from src.models.registry import load_model

from src.pipelines.inference_pipeline import (
    prepare_inference_features,
)

# ==========================================================
# Helpers
# ==========================================================

from ui.components.helpers import (
    load_threshold,
    format_risk_drivers,
)

# ==========================================================
# Sample Dataset Directory
# ==========================================================

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

# ==========================================================
# Session State
# ==========================================================

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table

def create_pdf(df):
    buffer = BytesIO()

    pdf = SimpleDocTemplate(buffer)

    data = [df.columns.tolist()] + df.values.tolist()

    table = Table(data)

    pdf.build([table])

    buffer.seek(0)

    return buffer


def initialize_session_state():
    """
    Initialize Streamlit Session State.
    """

    defaults = {

        "results": None,

        "threshold": None,

        "top_drivers": None,

        "source_name": None,

        "data_loaded": False,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# ==========================================================
# Batch Prediction
# ==========================================================

def score_batch(data_source):
    """
    Score customer records.
    """

    data = pd.read_csv(data_source)

    features = prepare_inference_features(data)

    model = load_model(MODEL_PATH)

    threshold = load_threshold()

    prediction_records = predict_churn(
        model,
        features,
        threshold=threshold,
    )

    append_prediction_log(
        prediction_records,
        PREDICTION_LOG_PATH,
        source="streamlit",
    )

    predictions = pd.DataFrame(
        prediction_records
    )

    reference_profile = load_profile(
        REFERENCE_PROFILE_PATH
    )

    batch_summary = summarize_batch(
        data,
        predictions,
        reference_profile,
    )

    save_profile(
        batch_summary,
        BATCH_MONITORING_PATH,
    )

    predictions["top_risk_drivers"] = (
        predictions["risk_drivers"]
        .apply(format_risk_drivers)
    )

    top_drivers = predictions[
        "risk_drivers"
    ].apply(

        lambda drivers:

        drivers[0]["display_name"]

        if drivers

        else "Unknown"
    )

    predictions = predictions.drop(
        columns=["risk_drivers"]
    )

    results = pd.concat(
        [
            data.reset_index(drop=True),
            predictions,
        ],
        axis=1,
    )

    return (
        results,
        threshold,
        top_drivers,
    )


# ==========================================================
# Run Prediction Once
# ==========================================================

def run_prediction(
    data_source,
    source_name,
):
    """
    Run prediction pipeline and
    store results in Session State.
    """

    (
        results,
        threshold,
        top_drivers,
    ) = score_batch(
        data_source
    )

    st.session_state.results = results

    st.session_state.threshold = threshold

    st.session_state.top_drivers = top_drivers

    st.session_state.source_name = source_name

    st.session_state.data_loaded = True

def render_scoring(metadata=None):
    """
    Render Batch Scoring page.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

    st.header("📈 Batch Scoring")

    st.caption(
        "Run predictions using one of the built-in demo datasets "
        "or upload your own customer records."
    )

    # Download Sample Dataset
    
    st.markdown("## 🚀 Enterprise Quick Start Workspace")

    st.caption(
        "Instantly experience the complete prediction workflow "
        "using one of the curated datasets below."
    )


    st.info(
    """
    ### 📋 Dataset Information

    • Three curated datasets are provided below for quick evaluation.

    • Each dataset is randomly sampled from the original Telco Customer Churn dataset.

    • A fixed random seed (`random_state = 42`) is used to ensures reproducibility.

    • All demo datasets are processed using the same prediction pipeline as user-uploaded customer records.

    • No separate demonstration model is used.

    • If you have your own dataset, Upload it to know the predictions.
    """
    )
   

    st.divider()

    # Cards

    col1, col2, col3 = st.columns(3)

    selected_dataset = None

    with col1:

        st.markdown("### ⚡ Quick Test")

        st.write("**10 Customers**")

        st.caption("Fast functionality validation")

        if st.button(
            "▶ Run Demo",
            key="demo10",
            width="stretch",
        ):
            selected_dataset = (
                SAMPLE_DIR / "sample_10.csv"
            )

    with col2:

        st.markdown("### ⭐ Standard Demo")

        st.write("**50 Customers**")

        st.caption("Ideal for presentations")

        if st.button(
            "▶ Run Demo",
            key="demo50",
            width="stretch",
        ):
            selected_dataset = (
                SAMPLE_DIR / "sample_50.csv"
            )

    with col3:

        st.markdown("### 🚀 Enterprise Demo")

        st.write("**100 Customers**")

        st.caption("Monitoring and analytics")

        if st.button(
            "▶ Run Demo",
            key="demo100",
            width="stretch",
        ):
            selected_dataset = (
                SAMPLE_DIR / "sample_100.csv"
            )

    st.divider()

    st.markdown("## 📂 Upload Your Own Customer Data for Predictions")

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
    )

    data_source = None

    if selected_dataset is not None:

        data_source = selected_dataset

    elif uploaded_file is not None:

        data_source = uploaded_file

    if data_source is None:

        st.info(
            "Choose one of the Enterprise Demo datasets "
            "or upload your own CSV."
        )

        return

    try:

        with st.spinner("Scoring customers..."):

            (
                results,
                threshold,
                top_drivers,
            ) = score_batch(data_source)

        # Temporary Debug Information

        st.subheader("🔍 Debug Information")

        st.write("Risk Band Distribution")
        st.write(results["risk_band"].value_counts())

        st.write("Top Risk Drivers")
        st.write(top_drivers.value_counts())

        st.success(
            "✅ Prediction completed successfully."
        )

        high_risk = int(
            (
                results["risk_band"] == "High"
            ).sum()
        )

        average_probability = float(
            results["churn_probability"].mean()
        )

        st.markdown("## Executive Risk Summary")

        c1, c2, c3, c4= st.columns(4)

        c1.metric(
            "Customers Analysed",
            len(results),
        )

        c2.metric(
            "High Risk",
            high_risk,
        )

        c3.metric(
            "Average Risk",
            f"{average_probability:.1%}",
        )

        c4.metric(
            "Threshold",
            f"{threshold:.2f}",
        )

        chart1, chart2 = st.columns(2)

        with chart1:

            st.subheader("Risk Distribution")

            st.write("Risk Band Distribution")
            st.write(results["risk_band"].value_counts())

            st.bar_chart(
                results["risk_band"]
                .value_counts()
            )

        with chart2:

            st.subheader("Top Risk Drivers")

            st.write("Top Risk Drivers")
            st.write(top_drivers.value_counts())
            
            st.bar_chart(
                top_drivers.value_counts()
            )

            st.write(results["risk_band"].value_counts())

        display_results = results

        st.subheader("Prediction Results")

        st.write(f"Rows displayed: {len(display_results)}")

        st.dataframe(
            display_results,
            width="stretch",
        )

        # ===========================
        # Download Predictions
        # ===========================

        st.subheader("Download Predictions")

        # CSV
        csv_data = display_results.to_csv(index=False).encode("utf-8")

        # JSON
        json_data = display_results.to_json(
            orient="records",
            indent=4,
        )

        # PDF
        pdf_data = create_pdf(display_results)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="📄 CSV",
                data=csv_data,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                label="📑 JSON",
                data=json_data,
                file_name="churn_predictions.json",
                mime="application/json",
            )

        with col3:
            st.download_button(
                label="📘 PDF",
                data=pdf_data,
                file_name="churn_predictions.pdf",
                mime="application/pdf",
            )

            st.download_button(
                label="📥 Download Predictions",
                data=results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:

        st.error(
            f"Unable to score customer records.\n\n{exc}"
        )