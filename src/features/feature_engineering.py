# FEATURE ENGINEERING   --> src/features/build_features.py

def build_features(df):
    X = df.drop("Churn Label", axis=1)
    y = df["Churn Label"]
    
    return X, y