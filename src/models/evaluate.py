import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
)

from src.config.settings import DEFAULT_THRESHOLD, POSITIVE_CLASS


def get_positive_probabilities(model, X):
    probabilities = model.predict_proba(X)
    positive_class_index = list(model.classes_).index(POSITIVE_CLASS)
    return probabilities[:, positive_class_index]


def labels_from_threshold(probabilities, threshold: float):
    return [POSITIVE_CLASS if probability >= threshold else "No" for probability in probabilities]


def binary_target(y):
    return pd.Series(y).reset_index(drop=True).eq(POSITIVE_CLASS).astype(int)


def evaluate(model, X, y, threshold: float = DEFAULT_THRESHOLD):
    positive_probabilities = get_positive_probabilities(model, X)
    predictions = labels_from_threshold(positive_probabilities, threshold)
    y_binary = binary_target(y)

    return {
        "roc_auc": float(roc_auc_score(y, positive_probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, pos_label=POSITIVE_CLASS, zero_division=0)),
        "brier_score": float(brier_score_loss(y_binary, positive_probabilities)),
    }


def tune_threshold(model, X, y, metric: str = "f1"):
    positive_probabilities = get_positive_probabilities(model, X)
    best_threshold = DEFAULT_THRESHOLD
    best_metrics = evaluate(model, X, y, threshold=best_threshold)
    best_score = best_metrics[metric]

    for threshold in [round(value / 100, 2) for value in range(10, 91, 5)]:
        metrics = evaluate(model, X, y, threshold=threshold)
        if metrics[metric] > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = metrics[metric]

    return best_threshold, best_metrics


def save_calibration_curve(y, positive_probabilities, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    y_binary = binary_target(y)
    prob_true, prob_pred = calibration_curve(
        y_binary,
        positive_probabilities,
        n_bins=10,
        pos_label=1,
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="Enterprise risk model")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Observed churn rate")
    plt.title("Probability Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_curve(y, positive_probabilities, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    display = RocCurveDisplay.from_predictions(
        y,
        positive_probabilities,
        pos_label=POSITIVE_CLASS,
        name="Enterprise risk model",
    )
    display.ax_.set_title("ROC Curve")
    display.ax_.grid(True, alpha=0.3)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(display.figure_)


def save_precision_recall_curve(y, positive_probabilities, path):
    """
    Save the Precision–Recall curve.

    Parameters
    ----------
    y : array-like
        Ground-truth labels.

    positive_probabilities : ndarray
        Predicted probabilities for the positive class.

    path : pathlib.Path
        Output image path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    display = PrecisionRecallDisplay.from_predictions(
        y,
        positive_probabilities,
        pos_label=POSITIVE_CLASS,
        name="Enterprise risk model",
    )

    ap = average_precision_score(
        binary_target(y),
        positive_probabilities,
    )

    display.ax_.set_title(
        f"Precision–Recall Curve (AP = {ap:.3f})"
    )
    display.ax_.grid(True, alpha=0.3)

    display.figure_.tight_layout()
    display.figure_.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(display.figure_)