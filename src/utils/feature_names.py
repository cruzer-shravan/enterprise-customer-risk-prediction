from src.config.settings import FEATURE_DISPLAY_NAMES


def format_feature_name(name: str) -> str:
    """
    Convert sklearn feature names into business-friendly names.
    """

    name = name.replace("num__", "")
    name = name.replace("cat__", "")

    parts = name.split("_")

    feature = parts[0]

    pretty = FEATURE_DISPLAY_NAMES.get(feature, feature)

    if len(parts) > 1:
        pretty += f": {' '.join(parts[1:])}"

    return pretty


def format_feature_names(feature_names):
    return [format_feature_name(name) for name in feature_names]