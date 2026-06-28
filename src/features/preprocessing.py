def preprocess(df):
    df = df.copy()

    if "Total Charges" in df.columns:
        df["Total Charges"] = df["Total Charges"].replace(" ", None)
        df["Total Charges"] = df["Total Charges"].astype(float)

    return df
