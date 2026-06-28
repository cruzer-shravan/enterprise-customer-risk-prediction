import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config.settings import FIGURES_DIR, POSITIVE_CLASS, TARGET_COLUMN
from src.models.evaluate import save_calibration_curve, save_roc_curve, save_precision_recall_curve
from src.models.explain import model_feature_importance
from src.utils.feature_names import (
    format_feature_name,
    format_feature_names,
)

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def confusion_matrix_report(y_true, predictions):
    labels = ["No", POSITIVE_CLASS]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=labels).ravel()
    return {
        "labels": labels,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def feature_importance_report(model, top_n=15):
    importances = model_feature_importance(model)
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        {
            "feature": format_feature_name(feature),
            "importance": round(importance, 6),
            "importance_percent": round(importance * 100, 2),
        }
        for feature, importance in ranked
    ]


def _save_model_comparison(comparison):
    comparison_rows = [
        {
            "model": model_name.replace("_", " ").title(),
            "F1": values["tuned_metrics"]["f1"],
            "Recall": values["tuned_metrics"]["recall"],
            "ROC-AUC": values["tuned_metrics"]["roc_auc"],
        }
        for model_name, values in comparison.items()
    ]
    comparison_df = pd.DataFrame(comparison_rows).set_index("model")
    ax = comparison_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=35)
    ax.set_title("Machine Learning Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_feature_importance(feature_importances):
    if not feature_importances:
        return
    feature_df = pd.DataFrame(feature_importances).sort_values("importance")
    ax = feature_df.plot(
        kind="barh",
        x="feature",
        y="importance_percent",
        figsize=(10, 7),
        color="#2F6F73",
        legend=False,
    )
    ax.set_title("Top Churn Risk Drivers")
    ax.set_xlabel("Relative importance (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "top_churn_risk_drivers.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_confusion_matrix(y_true, predictions):
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=["No", POSITIVE_CLASS],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(display.figure_)


def _save_churn_distribution(training_df):
    if training_df is None or TARGET_COLUMN not in training_df:
        return
    counts = training_df[TARGET_COLUMN].value_counts().reindex(["No", POSITIVE_CLASS]).fillna(0)
    ax = counts.plot(kind="bar", color=["#4C78A8", "#D65F5F"], figsize=(7, 5), rot=0)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn label")
    ax.set_ylabel("Customers")
    ax.bar_label(ax.containers[0], fmt="%.0f")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_business_impact_chart(business_impact):
    scenarios = (business_impact or {}).get("campaign_scenarios", [])
    if not scenarios:
        return
    scenario_df = pd.DataFrame(scenarios)
    scenario_df["target_rate"] = (scenario_df["target_rate"] * 100).round(0).astype(int).astype(str) + "%"
    ax = scenario_df.plot(
        kind="bar",
        x="target_rate",
        y=["expected_saved_value", "campaign_cost", "estimated_net_value"],
        figsize=(9, 5),
        color=["#4C78A8", "#F2A541", "#2F855A"],
    )
    ax.set_title("Retention Campaign Business Impact")
    ax.set_xlabel("Targeted customer share")
    ax.set_ylabel("Estimated value")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "business_impact.png", dpi=300, bbox_inches="tight")
    plt.close()


def _draw_box(ax, xy, text, width=1.8, height=0.72, color="#EAF4F4"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.2,
        edgecolor="#2F6F73",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        wrap=True,
    )


def _arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#334155"},
    )


def save_project_diagrams():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    boxes = [
        ((0.2, 2.4), "Customer\nDataset"),
        ((2.1, 2.4), "Validation +\nPreprocessing"),
        ((4.0, 2.4), "Feature\nEngineering"),
        ((5.9, 2.4), "Model Training\n+ Selection"),
        ((7.8, 2.4), "Explainability\n+ Reports"),
        ((9.7, 2.4), "FastAPI +\nStreamlit"),
        ((5.9, 0.9), "Model Registry\n+ Monitoring"),
    ]
    for xy, text in boxes:
        _draw_box(ax, xy, text)
    for x in [1.98, 3.88, 5.78, 7.68, 9.58]:
        _arrow(ax, (x, 2.76), (x + 0.22, 2.76))
    _arrow(ax, (6.8, 2.38), (6.8, 1.65))
    _arrow(ax, (7.65, 1.28), (9.7, 2.38))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0.4, 3.8)
    ax.set_title("Enterprise Customer Risk Prediction System Architecture", fontsize=14, weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lifecycle.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    flow_boxes = [
        ((0.4, 3.6), "Raw CSV\nInput"),
        ((2.6, 3.6), "Cleaned\nData"),
        ((4.8, 3.6), "Feature\nMatrix"),
        ((7.0, 3.6), "Trained\nModel"),
        ((7.0, 1.8), "Prediction\nScores"),
        ((4.8, 1.8), "Risk Bands +\nDrivers"),
        ((2.6, 1.8), "Dashboard +\nAPI Output"),
    ]
    for xy, text in flow_boxes:
        _draw_box(ax, xy, text, width=1.65, color="#F5F7FA")
    _arrow(ax, (2.05, 3.96), (2.6, 3.96))
    _arrow(ax, (4.25, 3.96), (4.8, 3.96))
    _arrow(ax, (6.45, 3.96), (7.0, 3.96))
    _arrow(ax, (7.82, 3.55), (7.82, 2.52))
    _arrow(ax, (7.0, 2.16), (6.45, 2.16))
    _arrow(ax, (4.8, 2.16), (4.25, 2.16))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(1.0, 4.8)
    ax.set_title("Data Flow Diagram", fontsize=14, weight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "data_flow_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figure_manifest():
    manifest = {
        "architecture_workflow": "reports/figures/lifecycle.png",
        "data_flow_diagram": "reports/figures/data_flow_diagram.png",
        "model_comparison": "reports/figures/model_comparison.png",
        "confusion_matrix": "reports/figures/confusion_matrix.png",
        "calibration_curve": "reports/figures/calibration_curve.png",
        "roc_curve": "reports/figures/roc_curve.png",
        "precision_recall_curve": "reports/figures/precision_recall_curve.png",
        "feature_importance": "reports/figures/feature_importance.png",
        "top_churn_risk_drivers": "reports/figures/top_churn_risk_drivers.png",
        "churn_distribution": "reports/figures/churn_distribution.png",
        "business_impact": "reports/figures/business_impact.png",
        "shap_summary": "reports/figures/shap_summary.png",
        "shap_waterfall": "reports/figures/shap_customer_0.png",
    }
    save_json(manifest, FIGURES_DIR / "figure_manifest.json")


def save_training_figures(
    comparison,
    feature_importances,
    y_true,
    predictions,
    positive_probabilities=None,
    training_df=None,
    business_impact=None,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_model_comparison(comparison)
    _save_feature_importance(feature_importances)
    _save_confusion_matrix(y_true, predictions)
    _save_churn_distribution(training_df)
    _save_business_impact_chart(business_impact)
    save_project_diagrams()

    if positive_probabilities is not None:
        save_calibration_curve(y_true, positive_probabilities, FIGURES_DIR / "calibration_curve.png")
        save_roc_curve(y_true, positive_probabilities, FIGURES_DIR / "roc_curve.png")
        save_precision_recall_curve(y_true, positive_probabilities, FIGURES_DIR / "precision_recall_curve.png")
    save_figure_manifest()
