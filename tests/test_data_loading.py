from pathlib import Path

from src.data.load_data import load_data


def test_load_data_reads_csv_fixture():
    path = Path("tests/fixtures/sample_churn.csv")

    df = load_data(path)

    assert df.shape[0] == 4
    assert "Churn Label" in df.columns
    assert set(df["Churn Label"]) == {"Yes", "No"}
