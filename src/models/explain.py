import numpy as np

from src.utils.feature_names import format_feature_name


def _unwrap_classifier(classifier):
    if hasattr(classifier, "feature_importances_") or hasattr(classifier, "coef_"):
        return classifier
    return getattr(classifier, "estimator_", classifier)


def _feature_names_from_transformer(transformer, columns):
    if hasattr(transformer, "named_steps"):
        final_step = list(transformer.named_steps.values())[-1]
        if hasattr(final_step, "categories_"):
            names = []
            for column, categories in zip(columns, final_step.categories_):
                names.extend([column] * len(categories))
            return names

    return list(columns)


def transformed_feature_sources(model):
    preprocessor = model.named_steps.get("preprocessor")
    sources = []

    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            sources.extend(list(columns))
            continue
        sources.extend(_feature_names_from_transformer(transformer, list(columns)))

    return sources


def model_feature_importance(model):
    classifier = _unwrap_classifier(model.named_steps.get("classifier"))

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()
    else:
        return {}

    sources = transformed_feature_sources(model)
    aggregated = {}
    for source, importance in zip(sources, importances):
        aggregated[source] = aggregated.get(source, 0.0) + float(abs(importance))

    total = sum(aggregated.values())
    if total <= 0:
        fallback_weight = 1.0 / len(aggregated) if aggregated else 0.0
        return {feature: fallback_weight for feature in aggregated}

    return {feature: importance / total for feature, importance in aggregated.items()}


def _fallback_importance(row):
    if row.empty:
        return {}

    fallback_weight = 1.0 / len(row)
    return {feature: fallback_weight for feature in row.index}


def explain_prediction(model, row, top_n: int = 5):
    importances = model_feature_importance(model)
    if not importances:
        importances = _fallback_importance(row)

    ranked_features = sorted(importances, key=importances.get, reverse=True)[:top_n]

    drivers = []
    for feature in ranked_features:
        value = row.get(feature)
        drivers.append(
            {
                "feature": feature,
                "display_name": format_feature_name(feature),
                "value": None if value is None else str(value),
                "importance": round(importances[feature], 6),
                "importance_percent": round(importances[feature] * 100, 2),
            }
        )

    return drivers
