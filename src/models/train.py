from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from src.config.settings import RANDOM_STATE
from src.explainability.shap_utils import ShapExplainer


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded_y)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        encoded_predictions = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(encoded_predictions)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def build_preprocessor(X, scale_numeric: bool = False):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def build_model_pipeline(X, estimator, scale_numeric: bool = False):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
            ("classifier", estimator),
        ]
    )


def get_model_candidates(X):
    candidates = {
        "logistic_regression": build_model_pipeline(
            X,
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        ),
        "random_forest": build_model_pipeline(
            X,
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
        "extra_trees": build_model_pipeline(
            X,
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
        "gradient_boosting": build_model_pipeline(
            X,
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
        "hist_gradient_boosting": build_model_pipeline(
            X,
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
        "adaboost": build_model_pipeline(
            X,
            AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
        "svc_rbf": build_model_pipeline(
            X,
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        ),
        "mlp_neural_network": build_model_pipeline(
            X,
            LabelEncodedClassifier(
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=300,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                )
            ),
            scale_numeric=True,
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = build_model_pipeline(
            X,
            LabelEncodedClassifier(
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                )
            ),
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = build_model_pipeline(
            X,
            LabelEncodedClassifier(
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                )
            ),
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        candidates["catboost"] = build_model_pipeline(
            X,
            LabelEncodedClassifier(
                CatBoostClassifier(
                    iterations=300,
                    depth=4,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="F1",
                    auto_class_weights="Balanced",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                )
            ),
        )
    except ImportError:
        pass

    return candidates


def train_model(X, y, model_name: str = "random_forest"):
    candidates = get_model_candidates(X)
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(candidates)}")

    model = candidates[model_name]

    model.fit(X, y)
    return model
