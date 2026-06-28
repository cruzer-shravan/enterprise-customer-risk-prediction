"""
monitoring.py

Model Monitoring page for the
Enterprise Customer Risk Prediction System.
"""

import pandas as pd
import streamlit as st

from src.config.settings import (
    BATCH_MONITORING_PATH,
    FEATURE_DISPLAY_NAMES,
    PREDICTION_LOG_PATH,
)

from ui.components.helpers import load_json


def render_monitoring(metadata=None):
    """
    Render Model Monitoring page.
    """

    st.header("📊 Model Monitoring")

    st.caption(
        "Monitor the deployed machine learning model, "
        "its performance metrics, calibration quality, "
        "and training metadata."
    )

    latest_batch = load_json(BATCH_MONITORING_PATH)

    prediction_log_exists = PREDICTION_LOG_PATH.exists()

    if latest_batch is None:

        st.info(
            "No monitoring information available yet.\n\n"
            "Run Batch Scoring to generate monitoring statistics."
        )

        return

    if metadata is None:
        st.write(metadata)
        st.error("Model metadata could not be loaded.")
        return


    # --------------------------------------------------
    # Prediction Summary
    # --------------------------------------------------

    with st.container():

        # --------------------------------------------------
        # Model Information
        # --------------------------------------------------

        st.subheader("🧠 Model Information")

        c1, c2, c3 = st.columns(3)

        st.json(metadata)
        
        c1.metric("Current Model", metadata["model_name"])
        c2.metric("Model Version", metadata["version"])
        c3.metric("Training Date", metadata["training_date"])

        c4, c5 = st.columns(2)

        c4.metric("Threshold", metadata["threshold"])
        c5.metric("Status", "Production")

        st.divider()

        # --------------------------------------------------
        # Performance Metrics
        # --------------------------------------------------

        st.subheader("📈 Performance Metrics")

        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{metadata['accuracy']:.1%}")
        c2.metric("ROC AUC", f"{metadata['roc_auc']:.1%}")
        c3.metric("Precision", f"{metadata['precision']:.1%}")

        c4, c5, c6 = st.columns(3)

        c4.metric("Recall", f"{metadata['recall']:.1%}")
        c5.metric("F1 Score", f"{metadata['f1']:.1%}")
        c6.metric("Brier Score", f"{metadata['brier_score']:.3f}")

        st.divider()

        # --------------------------------------------------
        # Calibration Status
        # --------------------------------------------------

        st.subheader("🎯 Calibration")

        if metadata["brier_score"] < 0.10:
            calibration_status = "Well Calibrated"
        elif metadata["brier_score"] < 0.20:
            calibration_status = "Acceptably Calibrated"
        else:
            calibration_status = "Poor Calibration"

        c1, c2 = st.columns(2)

        c1.metric(
            "Brier Score",
            f"{metadata['brier_score']:.3f}"
        )

        c2.metric(
            "Calibration Status",
            calibration_status
        )

        if calibration_status == "Well Calibrated":
            st.success("✅ Well Calibrated")
        elif calibration_status == "Acceptably Calibrated":
            st.warning("⚠ Acceptably Calibrated")
        else:
            st.error("❌ Poor Calibration")

        st.divider()

        # --------------------------------------------------
        # Dataset Information
        # --------------------------------------------------
        
        st.subheader("🗂 Dataset Information")

        c1, c2 = st.columns(2)

        c1.metric("Dataset Size", metadata["dataset_size"])
        c1.metric("Training Samples", metadata["train_size"])

        c2.metric("Test Samples", metadata["test_size"])
        c2.metric("Number of Features", metadata["num_features"])

        st.divider()

        # --------------------------------------------------
        # Latest Batch Summary
        # --------------------------------------------------

        st.subheader("📈 Latest Batch Summary")

        prediction_summary = latest_batch.get(
            "prediction",
            {},
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows Scored",
            prediction_summary.get(
                "row_count",
                0,
            ),
        )

        c2.metric(
            "Average Churn Probability",
            f"{prediction_summary.get('average_churn_probability',0):.1%}",
        )

        c3.metric(
            "Prediction Log",
            "Available"
            if prediction_log_exists
            else "Unavailable",
        )

        st.divider()

        # --------------------------------------------------
        # Risk Distribution
        # --------------------------------------------------

        st.subheader("🚨 Risk Distribution")

        risk_distribution = prediction_summary.get(
            "risk_band_distribution",
            {},
        )

        if risk_distribution:

            st.bar_chart(
                pd.Series(risk_distribution)
            )

        else:

            st.info(
                "Risk distribution is unavailable."
            )

        st.divider()

        # --------------------------------------------------
        # Drift Detection
        # --------------------------------------------------

        drift = latest_batch.get(
            "drift",
            {},
        )

        if not drift:

            st.info(
                "No drift analysis available."
            )

            return

        # --------------------------------------------------
        # Numeric Feature Drift
        # --------------------------------------------------

        st.markdown("### 📉 Numeric Feature Drift")

        numeric_df = pd.DataFrame(
            [
                drift.get(
                    "numeric_mean_shift",
                    {},
                )
            ]
        )

        numeric_df.rename(
            columns=FEATURE_DISPLAY_NAMES,
            inplace=True,
        )

        st.dataframe(
            numeric_df,
            width="stretch",
        )

        st.divider()

        # --------------------------------------------------
        # Categorical Feature Drift
        # --------------------------------------------------

        st.markdown(
            "### 🔄 Categorical Feature Drift"
        )

        categorical_df = pd.DataFrame(
            drift.get(
                "categorical_top_change",
                {},
            )
        ).T

        categorical_df.index = [
            FEATURE_DISPLAY_NAMES.get(
                feature,
                feature,
            )
            for feature in categorical_df.index
        ]

        st.dataframe(
            categorical_df,
            width="stretch",
        )