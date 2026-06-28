"""
streamlit_app.py

Main entry point for the Enterprise Customer Risk Prediction System.
"""

import sys
from pathlib import Path

import streamlit as st


# Add project root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# UI Components

from ui.components.styles import apply_custom_css
from ui.components.header import render_header
from ui.components.footer import render_footer
from ui.components.sidebar import render_sidebar
from ui.components.helpers import load_model_metadata




# Configure page

st.set_page_config(
    page_title="Enterprise Customer Risk Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    # Apply CSS
    apply_custom_css()
    
    # Load metadata
    metadata = load_model_metadata()
    
    # Render UI
    render_header()
    render_sidebar(metadata)
    st.title("Enterprise Customer Risk Prediction System")

    st.markdown("""
    Welcome to the Enterprise Customer Risk Prediction System.

    Use the navigation panel on the left to access:

    - Dashboard
    - Batch Scoring
    - Explainability
    - Monitoring
    - About
    """)
    render_footer()

if __name__ == "__main__":
    main()