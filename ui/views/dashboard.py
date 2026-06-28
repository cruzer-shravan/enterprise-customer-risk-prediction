"""
dashboard.py
Dashboard page.
"""

import pandas as pd
import streamlit as st

from src.config.settings import (
    BUSINESS_IMPACT_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    FEATURE_DISPLAY_NAMES,
)

from ui.components.helpers import (
    load_json,
)

from ui.components.metrics import (
    display_model_metrics,
)


def render_dashboard(metadata: dict):
    """
    Render dashboard page.
    """
    metadata = metadata or {}

    st.header("📊 Model Dashboard")

    metrics = load_json(METRICS_PATH) or {}
    feature_importance = load_json(FEATURE_IMPORTANCE_PATH) or []
    business_impact = load_json(BUSINESS_IMPACT_PATH) or {}

    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    with st.container():
        
        st.subheader("Deployment Information")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Deployment Model",
            metadata.get("deployment_model", "N/A"),
        )

        c2.metric(
            "Version",
            metadata.get("model_version", "N/A"),
        )

        threshold = metadata.get("threshold")

        c3.metric(
            "Decision Threshold",
            f"{threshold:.2f}" if threshold is not None else "N/A",
        )

    st.divider()

    # --------------------------------------------------
    # Model Performance
    # --------------------------------------------------

    with st.container():
        st.subheader("Model Performance")

        if metrics:
            display_model_metrics(metrics)
        else:
            st.warning("Metrics not available.")

    st.divider()

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    with st.container():

        st.subheader("Top Risk Drivers")

        if feature_importance:

            importance_df = (
                pd.DataFrame(feature_importance)
                .sort_values(
                    by="importance_percent",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            importance_df["display_name"] = (
                importance_df["feature"]
                .map(FEATURE_DISPLAY_NAMES)
                .fillna(importance_df["feature"])
            )

            st.bar_chart(
                importance_df.set_index("display_name")[
                    "importance_percent"
                ],
                width="stretch",
            )

        else:
            st.warning("Feature importance unavailable.")

    st.divider()
    
    # --------------------------------------------------
    # Business Impact
    # --------------------------------------------------

    with st.container():

        st.subheader("Retention Campaign Simulation")

        scenarios = business_impact.get(
            "campaign_scenarios",
            [],
        )

        if scenarios:

            st.dataframe(
                pd.DataFrame(scenarios),
                width="stretch",
            )

        else:

            st.info(
                "Business impact analysis unavailable."
            )