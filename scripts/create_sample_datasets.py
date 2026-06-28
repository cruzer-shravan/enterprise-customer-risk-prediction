"""
Create sample datasets for Batch Scoring.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_DATA)

# Sample size : Random seed
sample_configs = {
    10: 11,
    50: 42,
    100: 99,
}

for size, seed in sample_configs.items():

    sample_df = df.sample(
        n=size,
        random_state=seed,
    )

    sample_df.to_csv(
        OUTPUT_DIR / f"sample_{size}.csv",
        index=False,
    )

print("✅ Sample datasets created successfully.")