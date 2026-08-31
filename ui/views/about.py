"""
about.py

About page for the Enterprise Customer Risk Prediction System.
"""

import streamlit as st


def render_about(metadata=None):
    """
    Render About page.
    """

    st.header("ℹ About the Project")

    st.markdown(
        """
## Enterprise Customer Risk Prediction System

The **Enterprise Customer Risk Prediction System** is an end-to-end
Machine Learning application designed to predict customer churn risk
using historical customer information.

The application demonstrates a complete ML lifecycle including:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Explainable AI (SHAP)
- Batch prediction
- Model monitoring
- Interactive dashboard
- Enterprise deployment using Streamlit and FastAPI

---
"""
    )

    # --------------------------------------------------------
    # Project Overview
    # --------------------------------------------------------

    st.subheader("📌 Project Overview")

    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:

        st.markdown(
            """
### Objective

Predict customer churn risk and enable organizations to identify
customers likely to discontinue services.

The prediction enables proactive retention campaigns and improves
customer lifetime value.
"""
        )

    with overview_col2:

        st.markdown(
            """
### Business Value

- Reduce customer churn
- Improve customer retention
- Support business decision making
- Increase customer lifetime value
- Improve marketing effectiveness
"""
        )

    st.divider()

    # --------------------------------------------------------
    # Technology Stack
    # --------------------------------------------------------

    st.subheader("🛠 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown(
            """
### Machine Learning

- Python
- Scikit-learn
- SHAP
- Pandas
- NumPy
"""
        )

    with tech2:

        st.markdown(
            """
### Deployment

- Streamlit
- FastAPI
- Docker
- Uvicorn
"""
        )

    with tech3:

        st.markdown(
            """
### Development

- VS Code
- Git
- GitHub
- MLflow
"""
        )

    st.divider()

    # --------------------------------------------------------
    # Workflow
    # --------------------------------------------------------

    st.subheader("⚙ System Workflow")

    st.markdown(
        """
```text
Customer Dataset
        │
        ▼
Data Validation
        │
        ▼
Feature Engineering
        │
        ▼
Model Prediction
        │
        ▼
Risk Classification
        │
        ▼
SHAP Explainability
        │
        ▼
Monitoring & Reporting
"""
)

    st.divider()

    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    if metadata:

        st.subheader("📈 Current Deployment")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Deployment Model",
            metadata.get(
                "deployment_model",
                "N/A",
            ),
        )

        c2.metric(
            "Version",
            metadata.get(
                "model_version",
                "N/A",
            ),
        )

        threshold = metadata.get("threshold")

        c3.metric(
            "Threshold",
            f"{threshold:.2f}"
            if threshold is not None
            else "N/A",
        )

    st.divider()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    st.subheader("✨ Key Features")

    st.markdown(
        """

✅ Customer Churn Prediction

✅ Batch Customer Scoring

✅ Explainable AI using SHAP

✅ Probability Calibration

✅ Model Monitoring

✅ Drift Detection

✅ Batch Prediction Reports

✅ Interactive Dashboard

✅ Enterprise Ready Architecture
"""
)

    st.divider()

    # --------------------------------------------------------
    # Developer
    # --------------------------------------------------------

    st.subheader("👨‍💻 Developer")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("**Project**")
        st.markdown("**Developed by**")

    with col2:
        st.write("Enterprise Customer Risk Prediction System")
        st.write("U. Shravan Kumar")

    st.success(
    "Thank you for exploring the Enterprise Customer Risk Prediction System."
    )