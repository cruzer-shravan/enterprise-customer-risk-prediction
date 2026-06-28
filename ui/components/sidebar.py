"""
sidebar.py
Application sidebar.
"""

import streamlit as st


def render_sidebar(metadata):

    st.sidebar.title("Enterprise ML Dashboard")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Deployment")

    st.sidebar.write(
        f"**Model:** {metadata.get('deployment_model', 'N/A')}"
    )

    st.sidebar.write(
        f"**Version:** {metadata.get('model_version', 'N/A')}"
    )

    st.sidebar.write(
        f"**Threshold:** {metadata.get('threshold', 'N/A')}"
    )

    st.sidebar.markdown("---")

    st.sidebar.info(
        "Upload a CSV file in the Batch Scoring tab to generate predictions."
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Enterprise Customer Risk Prediction System"
    )