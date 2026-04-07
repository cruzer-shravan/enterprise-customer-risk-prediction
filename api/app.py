# api/app.py

from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API"}

@app.post("/predict")
def predict(data: dict):
    # dummy logic
    return {"prediction": "No"}