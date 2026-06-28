"""
Common layout wrapper for all Streamlit pages.
"""

import streamlit as st

from ui.components.styles import apply_custom_css
from ui.components.header import render_header
from ui.components.footer import render_footer
from ui.components.sidebar import render_sidebar
from ui.components.helpers import load_model_metadata


def render_page(title: str, content_function,):
    """
    Render a complete page with common layout.
    """

    st.set_page_config(
        page_title=title,
        layout="wide",
    )

    apply_custom_css()

    metadata = load_model_metadata()

    render_header()

    render_sidebar(metadata)

    content_function(metadata)

    render_footer()