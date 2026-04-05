# EVALUATION -->    src/models/evaluate.py

from sklearn.metrics import roc_auc_score

def evaluate(model, X, y):
    preds = model.predict_proba(X)[:, 1]
    score = roc_auc_score(y, preds)
    return score