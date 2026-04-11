"""Data cleaning helpers for the Conservative Formula project."""

import pandas as pd


EXCHANGE_CODE_MAP = {
    "1": "NYSE",
    "1.0": "NYSE",
    "31.0": "NYSE",
    "2": "AMEX",
    "2.0": "AMEX",
    "32.0": "AMEX",
    "3": "NASDAQ",
    "3.0": "NASDAQ",
    "33.0": "NASDAQ",
    "31": "NYSE",
    "32": "AMEX",
    "33": "NASDAQ",
}

VALID_EXCHANGES = {"NYSE", "AMEX", "NASDAQ"}
RETURN_CAP = 5.0


def rename_and_parse_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the legacy notebook's date normalization step."""
    cleaned = frame.copy()
    if "date" in cleaned.columns and "Date" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"date": "Date"})
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    return cleaned


def filter_supported_exchanges(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only NYSE, AMEX, and NASDAQ observations."""
    cleaned = frame
    cleaned["EXCHCD"] = cleaned["EXCHCD"].astype(str).replace(EXCHANGE_CODE_MAP)
    return cleaned.loc[cleaned["EXCHCD"].isin(VALID_EXCHANGES)]


def clean_basic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's basic row-level cleaning steps."""
    cleaned = frame.copy()
    cleaned["ALTPRC"] = cleaned["ALTPRC"].abs()
    cleaned = cleaned.dropna(subset=["DLRET", "RETX", "ALTPRC", "SHROUT"], how="all")
    cleaned = cleaned.dropna(subset=["DLRET", "RETX"], how="all")
    return cleaned


def build_adjusted_return(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's delisting-adjusted return logic."""
    cleaned = frame.copy()

    cleaned.loc[cleaned["DLRET"].isnull(), "RET ADJ"] = cleaned["RET"]
    cleaned.loc[
        cleaned["DLRET"].notnull() & cleaned["DLSTCD"].notnull(),
        "RET ADJ",
    ] = cleaned["DLRET"]
    cleaned.loc[cleaned["DLSTCD"].between(551, 574), "RET ADJ"] = -0.3
    cleaned.loc[cleaned["DLSTCD"].isin([500, 520, 580, 584]), "RET ADJ"] = -0.3
    cleaned.loc[cleaned["DLSTCD"] == 100, "RET ADJ"] = cleaned["RET"]

    recognized_delisting = (
        cleaned["DLSTCD"].isnull()
        | (cleaned["DLSTCD"].notnull() & cleaned["DLRET"].notnull())
        | cleaned["DLSTCD"].between(551, 574)
        | cleaned["DLSTCD"].isin([500, 520, 580, 584, 100])
    )
    cleaned.loc[~recognized_delisting, "RET ADJ"] = -1

    cleaned.loc[
        cleaned["DLRET"].isin(["P", "S"])
        & pd.to_numeric(cleaned["RET"], errors="coerce").notna()
        & pd.to_numeric(cleaned["RET ADJ"], errors="coerce").isna(),
        "RET ADJ",
    ] = cleaned["RET"]

    cleaned["RET ADJ"] = pd.to_numeric(cleaned["RET ADJ"], errors="coerce")
    cleaned = cleaned.dropna(subset=["RET ADJ"])
    cleaned["RET ADJ"] = cleaned["RET ADJ"].clip(upper=RETURN_CAP)
    return cleaned


def add_market_cap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add market cap and lagged market cap exactly as in the legacy notebook."""
    cleaned = frame.copy()
    cleaned["MKT Cap"] = cleaned["ALTPRC"] * cleaned["SHROUT"]
    cleaned["Lagged MKT Cap"] = cleaned.groupby("PERMNO")["MKT Cap"].shift(1)
    return cleaned


def prepare_initial_stock_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the first end-to-end cleaning block from the legacy notebook."""
    cleaned = rename_and_parse_date(frame)
    cleaned = filter_supported_exchanges(cleaned)
    cleaned = clean_basic_columns(cleaned)
    cleaned = build_adjusted_return(cleaned)
    cleaned = add_market_cap_columns(cleaned)
    return cleaned

