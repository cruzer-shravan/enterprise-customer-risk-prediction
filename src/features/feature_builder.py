from src.config.settings import ID_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN


def build_features(df):
    drop_columns = [TARGET_COLUMN, *LEAKAGE_COLUMNS, *ID_COLUMNS]
    existing_drop_columns = [col for col in drop_columns if col in df.columns]

    X = df.drop(columns=existing_drop_columns)
    y = df[TARGET_COLUMN]

    return X, y
