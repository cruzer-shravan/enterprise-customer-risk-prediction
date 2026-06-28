from src.config.settings import ID_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN
from src.data.validate_data import validate_inference_data
from src.features.preprocessing import preprocess


def prepare_inference_features(df):
    df = preprocess(df)
    validate_inference_data(df)
    drop_columns = [TARGET_COLUMN, *LEAKAGE_COLUMNS, *ID_COLUMNS]
    existing_drop_columns = [col for col in drop_columns if col in df.columns]
    return df.drop(columns=existing_drop_columns)
