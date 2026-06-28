from pathlib import Path

import joblib


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_versioned_model(model, model_dir: Path, version: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = model_dir / f"model_{version}.pkl"
    joblib.dump(model, versioned_path)
    return versioned_path


def load_model(path: Path):
    return joblib.load(path)
