"""
footer.py
Application footer.
"""

import streamlit as st


def render_footer():

    st.divider()

    st.markdown(
        """
<div style='text-align:center;font-size:14px;color:gray;'>

Enterprise Customer Risk Prediction System

Developed by **U. Shravan Kumar**

© 2026

</div>
""",
        unsafe_allow_html=True,
    )