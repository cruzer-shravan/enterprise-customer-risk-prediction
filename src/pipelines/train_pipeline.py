import json
from datetime import datetime, timezone

import sklearn

from src.config.settings import (
    BUSINESS_IMPACT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    REFERENCE_PROFILE_PATH,
    MODEL_SELECTION_TOLERANCE,
)
from src.data.load_data import load_data
from src.data.split_data import split_data
from src.data.validate_data import validate_training_data
from src.explainability.shap_utils import ShapExplainer
from src.features.feature_builder import build_features
from src.features.preprocessing import preprocess
from src.models.business_impact import build_business_impact_report
from src.models.evaluate import evaluate, get_positive_probabilities, labels_from_threshold, tune_threshold
from src.models.reporting import (
    confusion_matrix_report,
    feature_importance_report,
    save_json,
    save_training_figures,
)
from src.models.registry import save_model, save_versioned_model
from src.models.train import get_model_candidates
from src.monitoring.drift_detection import build_reference_profile, save_profile

# ---------------------------------------------------------------------
# Model selection preferences
# ---------------------------------------------------------------------

TREE_MODELS = {
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
}


def select_deployment_model(comparison, tolerance):
    """
    Select the deployment model using an explainability-aware strategy.

    Selection Strategy
    ------------------
    1. Rank all candidate models by:
       - F1 Score (primary metric)
       - ROC-AUC (secondary metric)

    2. Identify every model whose F1 score is within the specified
       tolerance of the highest F1 score.

    3. Among those models, prefer an explainable tree-based model
       (Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting,
       XGBoost, LightGBM, or CatBoost) because these models provide:

       - Native feature importance
       - Fast SHAP TreeExplainer support
       - Better interpretability for business users
       - Easier production monitoring

    4. If no explainable tree-based model falls within the tolerance,
       deploy the highest-performing model regardless of model type.

    Parameters
    ----------
    comparison : dict
        Dictionary containing tuned evaluation metrics for each model.

    tolerance : float
        Maximum acceptable F1 score difference from the best-performing
        model when selecting an explainable alternative.

    Returns
    -------
    str
        Name of the model selected for deployment.
    """

    # Sort models by F1 score (descending), then ROC-AUC (descending)
    ranked = sorted(
        comparison.items(),
        key=lambda item: (
            item[1]["tuned_metrics"]["f1"],
            item[1]["tuned_metrics"]["roc_auc"],
        ),
        reverse=True,
    )

    # Highest-performing model before applying deployment policy
    best_name, best_result = ranked[0]
    best_f1 = best_result["tuned_metrics"]["f1"]

    # Find all tree-based models whose F1 score is within the tolerance
    # of the highest-performing model.
    explainable_candidates = [
        (name, result)
        for name, result in ranked
        if (
            best_f1 - result["tuned_metrics"]["f1"] <= tolerance
            and name in TREE_MODELS
        )
    ]

    # If one or more explainable models satisfy the tolerance criterion,
    # choose the highest-ranked explainable model.
    if explainable_candidates:

        selected_name, selected_result = explainable_candidates[0]

        print("\n" + "=" * 70)
        print("Explainability-aware Deployment Selection")
        print("=" * 70)
        print(f"Highest F1 Model      : {best_name}")
        print(f"Deployment Model        : {selected_name}")
        print(f"Highest F1            : {best_f1:.4f}")
        print(
            f"Selected F1           : "
            f"{selected_result['tuned_metrics']['f1']:.4f}"
        )
        print(
            f"Difference            : "
            f"{best_f1 - selected_result['tuned_metrics']['f1']:.4f}"
        )
        print(
            "Reason                : "
            "Tree-based model selected because it is within the "
            "configured tolerance and provides better explainability."
        )
        print("=" * 70 + "\n")

        return best_name, selected_name

    # If no explainable alternative is available, 
    # deploy the highest-performing model.
    print("\n" + "=" * 70)
    print("Deployment Selection")
    print("=" * 70)
    print("No explainable tree-based model found within tolerance.")
    print(f"Deployment Model : {best_name}")
    print("=" * 70 + "\n")

    return best_name, best_name


def log_to_mlflow(model_name, model, metrics, threshold):
    try:
        import mlflow
    except ImportError:
        return

    try:
        mlflow.set_experiment("enterprise-customer-risk-prediction")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({"model": model_name, "target": "Churn Label", "threshold": threshold})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    except Exception as exc:
        print(f"MLflow logging skipped for {model_name}: {exc}")


def _sample_rows(matrix, limit=200):
    return matrix[: min(limit, matrix.shape[0])]


def save_shap_figures(best_model, X_train, X_test):
    try:
        preprocessor = best_model.named_steps["preprocessor"]
        classifier = best_model.named_steps["classifier"]

        if classifier.__class__.__name__ == "LabelEncodedClassifier":
            classifier = classifier.estimator_

        X_train_processed = _sample_rows(
            preprocessor.transform(X_train), limit=200
        )
        X_test_processed = _sample_rows(
            preprocessor.transform(X_test), limit=200
        )

        feature_names = preprocessor.get_feature_names_out()

        explainer = ShapExplainer(
            model=classifier,
            X_train=X_train_processed,
            feature_names=feature_names,
        )

        explainer.generate_summary_plot(
            X_test_processed,
            save_path=FIGURES_DIR / "shap_summary.png",
        )

        explainer.generate_waterfall_plot(
            X_test_processed,
            index=0,
            save_path=FIGURES_DIR / "shap_customer_0.png",
        )

    except ValueError as exc:
        print(f"SHAP not available for this model: {exc}")

    except Exception:
        raise


def run_pipeline():
    run_timestamp = datetime.now(timezone.utc)
    model_version = run_timestamp.strftime("%Y%m%d%H%M%S")
    df = load_data(RAW_DATA_PATH)
    df = preprocess(df)
    validate_training_data(df)
    save_profile(build_reference_profile(df), REFERENCE_PROFILE_PATH)
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    comparison = {}
    trained_models = {}

    for model_name, model in get_model_candidates(X_train).items():
        model.fit(X_train, y_train)
        default_metrics = evaluate(model, X_test, y_test)
        best_threshold, tuned_metrics = tune_threshold(model, X_test, y_test)

        comparison[model_name] = {
            "default_threshold": 0.5,
            "default_metrics": default_metrics,
            "best_threshold": best_threshold,
            "tuned_metrics": tuned_metrics,
        }
        trained_models[model_name] = model
        log_to_mlflow(model_name, model, tuned_metrics, best_threshold)

    highest_f1_model_name, deployment_model_name = select_deployment_model(
    comparison, tolerance=MODEL_SELECTION_TOLERANCE,
    )

    best_model = trained_models[deployment_model_name]
    best_threshold = comparison[deployment_model_name]["best_threshold"]
    positive_probabilities = get_positive_probabilities(best_model, X_test)
    threshold_predictions = labels_from_threshold(positive_probabilities, best_threshold)

    metrics = {
        "highest_f1_model": highest_f1_model_name,
        "deployment_model": deployment_model_name,
        "selection_strategy": "Explainability-aware",
        "selection_tolerance": MODEL_SELECTION_TOLERANCE,
        "threshold": best_threshold,
        **comparison[deployment_model_name]["tuned_metrics"],
    }

    confusion_summary = confusion_matrix_report(y_test, threshold_predictions)
    feature_importances = feature_importance_report(best_model)
    business_impact = build_business_impact_report(X_test, y_test, positive_probabilities)

    save_shap_figures(best_model, X_train, X_test)

    save_model(best_model, MODEL_PATH)
    versioned_model_path = save_versioned_model(best_model, MODEL_PATH.parent, model_version)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, METRICS_PATH)
    save_json(comparison, MODEL_COMPARISON_PATH)
    save_json(confusion_summary, CONFUSION_MATRIX_PATH)
    save_json(feature_importances, FEATURE_IMPORTANCE_PATH)
    save_json(business_impact, BUSINESS_IMPACT_PATH)
    save_training_figures(
        comparison,
        feature_importances,
        y_test,
        threshold_predictions,
        positive_probabilities=positive_probabilities,
        training_df=df,
        business_impact=business_impact,
    )
    
    MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                # Enterprise Dashboard
                "model_name": deployment_model_name.replace("_", " ").title(),
                "version": model_version,
                "training_date": run_timestamp.strftime("%d-%b-%Y"),

                # Performance Metrics
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "threshold": best_threshold,
                "brier_score": metrics["brier_score"],

                # Dataset Information
                "dataset_size": int(len(df)),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "num_features": int(X.shape[1]),

                # Existing Metadata
                "model_version": model_version,
                "created_at": run_timestamp.isoformat(),

                "highest_f1_model": highest_f1_model_name,
                "deployment_model": deployment_model_name,

                "selection_strategy": "Explainability-aware",
                "selection_tolerance": MODEL_SELECTION_TOLERANCE,

                "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
                "versioned_model_path": str(
                    versioned_model_path.relative_to(PROJECT_ROOT)
                ),

                "training_rows": int(len(X)),
                "feature_count": int(X.shape[1]),

                "sklearn_version": sklearn.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Versioned model saved to: {versioned_model_path}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Model metadata saved to: {MODEL_METADATA_PATH}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_PATH}")
    print(f"Business impact saved to: {BUSINESS_IMPACT_PATH}")
    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")
    
    print("=" * 70)
    print("Training Pipeline Completed Successfully")
    print("=" * 70)
    print(f"Highest F1 Model      : {highest_f1_model_name}")
    print(f"Deployment Model      : {deployment_model_name}")
    print(f"Threshold             : {best_threshold:.2f}")
    print(f"ROC-AUC               : {metrics['roc_auc']:.4f}")
    print(f"Accuracy              : {metrics['accuracy']:.4f}")
    print(f"Precision             : {metrics['precision']:.4f}")
    print(f"Recall                : {metrics['recall']:.4f}")
    print(f"F1 Score              : {metrics['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

