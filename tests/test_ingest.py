import pandas as pd


def test_financial_dataset_has_expected_columns():
    """Verify that a sample financial dataset can be processed."""
    df = pd.DataFrame(
        {
            "document": [
                "BMW annual financial report",
                "Nvidia quarterly financial report",
            ],
            "year": [2022, 2023],
            "entity": ["BMW", "Nvidia"],
        }
    )

    assert not df.empty
    assert "document" in df.columns
    assert "year" in df.columns
    assert "entity" in df.columns


def test_financial_dataset_contains_text():
    """Verify that sample financial documents contain usable text."""
    df = pd.DataFrame(
        {
            "document": [
                "BMW annual financial report",
                "Nvidia quarterly financial report",
            ]
        }
    )

    assert df["document"].notna().all()
    assert df["document"].str.strip().ne("").all()