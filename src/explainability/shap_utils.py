from pathlib import Path

import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.utils.feature_names import format_feature_names

class ShapExplainer:
     
    def __init__(self, model, X_train, feature_names):
        self.model = model

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        self.X_train = X_train
        self.feature_names = format_feature_names(feature_names)
        
        for i, name in enumerate(self.feature_names):
            print(f"{i:2d} : {name}")

        self.explainer = self._build_explainer()

    def _build_explainer(self):

        tree_models = (
            RandomForestClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
        )

        try:
            from xgboost import XGBClassifier
            tree_models += (XGBClassifier,)
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier
            tree_models += (LGBMClassifier,)
        except ImportError:
            pass

        try:
            from catboost import CatBoostClassifier
            tree_models += (CatBoostClassifier,)
        except ImportError:
            pass

        if isinstance(self.model, tree_models):

            return shap.TreeExplainer(
                self.model,
                data=self.X_train,
                feature_perturbation="interventional",
            )

        if isinstance(self.model, LogisticRegression):

            return shap.LinearExplainer(
                self.model,
                self.X_train,
            )

        if isinstance(self.model, (SVC, MLPClassifier)):

            background = shap.sample(
                self.X_train,
                min(100, len(self.X_train)),
            )

            return shap.KernelExplainer(
                self.model.predict_proba,
                background,
            )

        raise ValueError(
            f"Unsupported model type: {type(self.model).__name__}"
        )
    

    def compute_shap_values(self, X_sample):

        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()

        explanation = self.explainer(X_sample)

        if isinstance(explanation, shap.Explanation):

            return shap.Explanation(
                values=explanation.values,
                base_values=explanation.base_values,
                data=explanation.data,
                feature_names=self.feature_names,
            )

        values = explanation

        if isinstance(values, list):
            values = values[1]

        return shap.Explanation(
            values=values,
            base_values=self.explainer.expected_value,
            data=X_sample,
            feature_names=self.feature_names,
        )
    def generate_summary_plot(self, X_sample, save_path=None):

        explanation = self.compute_shap_values(X_sample)

        # Select the positive class (Churn = Yes)
        if explanation.values.ndim == 3:
            explanation = explanation[:, :, 1]

        print(type(explanation))
        print(explanation.feature_names[:10])

        plt.figure(figsize=(13, 9))

        shap.summary_plot(
            explanation.values,
            explanation.data,
            feature_names=explanation.feature_names,
            max_display=20,
            show=False,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

    def generate_waterfall_plot(self, X_sample, index=0, save_path=None):
        """
        Generate a SHAP waterfall plot for one customer.

        The feature names are preserved so the plot displays
        readable feature names instead of "Feature 1", "Feature 2", etc.
        """

        explanation = self.compute_shap_values(X_sample)

        # Multi-output classifiers
        if len(explanation.values.shape) == 3:
            single = explanation[index, :, 1]
        else:
            single = explanation[index]

        plt.figure(figsize=(13, 9))

        shap.plots.waterfall(
            single,
            max_display=15,
            show=False,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()