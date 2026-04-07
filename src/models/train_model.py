# MODEL TRAINING -->  src/models/train_model.py

from src.models import train_model

from sklearn.ensemble import RandomForestClassifier

def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model