from sklearn.model_selection import train_test_split

from src.config.settings import RANDOM_STATE, TEST_SIZE


def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
