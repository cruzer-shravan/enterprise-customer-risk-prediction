# SAVE MODEL (PRODUCTION MUST)      -->    src/models/save_model.py

import joblib

def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)