# CREATE PIPELINE   --->    src/pipelines/train_pipeline.py

from src.data.load_data import load_data
from features.training_pipeline import preprocess
from features.feature_engineering import build_features
from src.models.train_model import train_model
from src.models.evaluate import evaluate

def run_pipeline():
    df = load_data("data/Telco_customer_churn.xlsx")
    
    df = preprocess(df)
    X, y = build_features(df)
    
    model = train_model(X, y)
    score = evaluate(model, X, y)
    
    print(f"ROC-AUC: {score}")

if __name__ == "__main__":
    run_pipeline()