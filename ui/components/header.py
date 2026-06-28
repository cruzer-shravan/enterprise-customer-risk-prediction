"""
header.py
Application header.
"""

import streamlit as st
from pathlib import Path


def render_header():
    """
    Render application header.
    """

    logo_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "assets"
        / "logo.png"
    )

    col1, col2 = st.columns([1, 8])

    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=80)

    with col2:
        st.title("Enterprise Customer Risk Prediction System")
        st.caption(
            "AI-powered Customer Churn Prediction using Explainable Machine Learning"
        )

    st.divider()