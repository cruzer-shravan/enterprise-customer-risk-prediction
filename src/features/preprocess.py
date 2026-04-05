# Preprocessing -->     src/features/preprocess.py

def preprocess(df):
    df = df.copy()
    
    # example
    df.dropna(inplace=True)
    
    return df