"""Data loading helpers for the Conservative Formula project."""

from pathlib import Path

import pandas as pd


def load_stock_data(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a raw stock panel from CSV or parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file format for {path}")
