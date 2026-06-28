"""
explainability.py

Model Explainability page for the
Enterprise Customer Risk Prediction System.
"""

from pathlib import Path

import streamlit as st

from src.config.settings import FIGURES_DIR



def render_explainability(metadata=None):
    """
    Render Explainability page.
    """

    st.header("🧠 Model Explainability")

    st.caption(
        "Understand how the machine learning model "
        "makes customer churn predictions using SHAP "
        "and probability calibration."
    )

    # --------------------------------------------------
    # Model Overview
    # --------------------------------------------------

    st.markdown("## Executive Model Summary")

    st.info(
        """
    This page explains how the trained machine learning model
    predicts customer churn.

    The visualizations below illustrate:

    • Which features have the greatest impact on predictions.

    • How those features influence individual customer risk.

    • Whether predicted probabilities are reliable.

    These explanations improve model transparency and help
    business stakeholders trust the model's recommendations.
    """
    )

    st.divider()



    # --------------------------------------------------
    # SHAP Explainability
    # --------------------------------------------------

    with st.container():

        st.markdown("## 🌍 Global Feature Importance")

        shap_summary_path = (
            FIGURES_DIR / "shap_summary.png"
        )

        if Path(shap_summary_path).exists():

            st.image(
                shap_summary_path,
                caption="Global Feature Impact on Customer Churn Prediction",
                width="stretch",
            )
            st.caption(
                "Features are ranked by their average impact on churn predictions."
            )

        else:

            st.info(
                "Global SHAP summary plot is not available."
            )

    st.divider()

    # --------------------------------------------------
    # Individual Customer SHAP
    # --------------------------------------------------

    with st.container():

        st.markdown("## 👤 Individual Customer Explanation")

        shap_customer_path = (
            FIGURES_DIR / "shap_customer_0.png"
        )

        if Path(shap_customer_path).exists():

            st.image(
                shap_customer_path,
                caption="Customer-Level SHAP Explanation",
                width="stretch",
            )

            st.caption(
                "Positive SHAP values increase churn risk while negative values reduce it."
            )

        else:

            st.info(
                "Customer SHAP explanation is not available."
            )

    st.divider()

    # --------------------------------------------------
    # Probability Calibration
    # --------------------------------------------------

    with st.container():

        st.markdown("### 📈 Probability Calibration")

        calibration_curve_path = (
            FIGURES_DIR / "calibration_curve.png"
        )

        if Path(calibration_curve_path).exists():

            st.image(
                calibration_curve_path,
                caption="Calibration Curve - Reliability of Predicted Churn Probabilities",
                width="stretch",
            )

            st.caption(
                "A calibration curve close to the diagonal indicates reliable probability estimates."
            )

        else:

            st.info(
                "Calibration curve is not available."
            )

    st.divider()

    # --------------------------------------------------
    # Business Interpretation
    # --------------------------------------------------

    with st.container():

        st.markdown("## 💡 Business Interpretation")

        st.success(
        """
        The model predicts customer churn by combining multiple
        customer characteristics rather than relying on a single feature.

        Typical factors increasing churn risk include:

        • Short customer tenure

        • High monthly charges

        • Month-to-month contracts

        Factors that often reduce churn risk include:

        • Longer contracts

        • Higher customer lifetime value

        • Stable customer relationships

        Business teams can use these insights to identify
        high-risk customers and implement targeted retention
        strategies before churn occurs.
        """
        )
    
    # --------------------------------------------------
    # Key Model Insights
    # -------------------------------------------------

    with st.container():

        st.markdown("## 5. Key Model Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Explainability Method",
                "SHAP"
            )

            st.metric(
                "Probability Calibration",
                "Enabled"
            )

        with col2:
            st.metric(
                "Feature Importance",
                "Global + Local"
            )

            st.metric(
                "Business Ready",
                "Yes"
            )