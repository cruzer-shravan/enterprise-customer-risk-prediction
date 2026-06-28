"""
styles.py
Applies custom CSS to the Streamlit application.
"""

from pathlib import Path
import streamlit as st


def apply_custom_css():
    """
    Load and apply the application's custom CSS.
    """

    css_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "assets"
        / "css.css"
    )

    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )