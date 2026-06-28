"""
metrics.py
Reusable metric components.
"""

import streamlit as st


def display_model_metrics(metrics):

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "ROC-AUC",
        f"{metrics.get('roc_auc',0):.3f}",
    )

    c2.metric(
        "Accuracy",
        f"{metrics.get('accuracy',0):.3f}",
    )

    c3.metric(
        "Precision",
        f"{metrics.get('precision',0):.3f}",
    )

    c4.metric(
        "Recall",
        f"{metrics.get('recall',0):.3f}",
    )

    c5.metric(
        "F1 Score",
        f"{metrics.get('f1',0):.3f}",
    )