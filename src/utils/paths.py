from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
