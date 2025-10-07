import pandas as pd
from pathlib import Path


# Default to repo references/ethnicity_mapping.csv
def load_eth_map(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve(
        ).parents[1] / "references" / "ethnicity_mapping.csv"
    df = pd.read_csv(path, dtype=str)
    df["ethnicity_code"] = df["ethnicity_code"].str.strip().str.upper()
    df["ethnicity_group"] = df["ethnicity_group"].str.strip()
    return dict(zip(df["ethnicity_code"], df["ethnicity_group"]))


def map_ethnicity_series(s, mapping: dict | None = None):
    if mapping is None:
        mapping = load_eth_map()
    s = s.astype("string").str.strip().str.upper()
    s = s.mask(s == "99", "99")
    return s.map(mapping).fillna("Not known").astype("category")
