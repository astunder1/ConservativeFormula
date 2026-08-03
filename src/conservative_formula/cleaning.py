"""Data cleaning helpers for the Conservative Formula project."""

import pandas as pd

EXCHANGE_CODE_MAP = {
    "1": "NYSE",
    "2": "AMEX",
    "3": "NASDAQ",
    "31": "NYSE",
    "32": "AMEX",
    "33": "NASDAQ",
}

VALID_EXCHANGES = {"NYSE", "AMEX", "NASDAQ"}
COMMON_SHARE_CODES = {10, 11}
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
    cleaned = frame.copy()
    cleaned = cleaned.sort_values(["PERMNO", "Date"])
    exchcd = pd.to_numeric(cleaned["EXCHCD"], errors="coerce").astype("Int64")
    exchcd = exchcd.groupby(cleaned["PERMNO"]).ffill()
    cleaned["EXCHCD"] = exchcd.astype(str).replace(EXCHANGE_CODE_MAP)
    return cleaned.loc[cleaned["EXCHCD"].isin(VALID_EXCHANGES)].copy()


def filter_common_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only ordinary common shares (CRSP SHRCD 10/11)."""
    cleaned = frame.copy()
    shrcd = pd.to_numeric(cleaned["SHRCD"], errors="coerce")
    return cleaned.loc[shrcd.isin(COMMON_SHARE_CODES)].copy()


def clean_basic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's basic row-level cleaning steps."""
    cleaned = frame.copy()
    cleaned["ALTPRC"] = cleaned["ALTPRC"].abs()
    cleaned = cleaned.dropna(subset=["DLRET", "RETX"], how="all")
    return cleaned


def build_adjusted_return(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's delisting-adjusted return logic."""
    cleaned = frame.copy()

    cleaned["RET"] = pd.to_numeric(cleaned["RET"], errors="coerce")
    cleaned["DLRET"] = pd.to_numeric(cleaned["DLRET"], errors="coerce")
    cleaned["DLSTCD"] = pd.to_numeric(cleaned["DLSTCD"], errors="coerce")

    performance_delist = cleaned["DLSTCD"].between(551, 574) | cleaned["DLSTCD"].isin(
        [500, 520, 580, 584]
    )

    # Default to the ordinary monthly return.
    cleaned["RET ADJ"] = cleaned["RET"]

    # Compound the ordinary monthly return with the CRSP delisting return when
    # both are present, rather than substituting one for the other.
    has_dlret = cleaned["DLRET"].notna()
    cleaned.loc[has_dlret, "RET ADJ"] = (
        (1.0 + cleaned.loc[has_dlret, "RET"].fillna(0.0)) * (1.0 + cleaned.loc[has_dlret, "DLRET"]) - 1.0
    )

    # Impute -30% (NYSE/AMEX) or -55% (NASDAQ) for selected performance-related
    # delistings when no usable delisting return is available.
    missing_dlret_performance = cleaned["DLRET"].isna() & performance_delist
    cleaned.loc[missing_dlret_performance & (cleaned["EXCHCD"] == "NASDAQ"), "RET ADJ"] = -0.55
    cleaned.loc[missing_dlret_performance & (cleaned["EXCHCD"] != "NASDAQ"), "RET ADJ"] = -0.3

    recognized_delisting = (
        cleaned["DLSTCD"].isna()
        | cleaned["DLRET"].notna()
        | performance_delist
        | (cleaned["DLSTCD"] == 100)
    )
    cleaned.loc[~recognized_delisting, "RET ADJ"] = -1.0

    cleaned = cleaned.dropna(subset=["RET ADJ"])
    cleaned["RET ADJ"] = cleaned["RET ADJ"].clip(lower=-1.0, upper=RETURN_CAP)
    return cleaned


def add_market_cap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add market cap and lagged market cap exactly as in the legacy notebook."""
    cleaned = frame.copy()
    cleaned = cleaned.sort_values(["PERMNO", "Date"])
    cleaned["MKT Cap"] = cleaned["ALTPRC"] * cleaned["SHROUT"]
    cleaned["Lagged MKT Cap"] = cleaned.groupby("PERMNO")["MKT Cap"].shift(1)
    return cleaned


def filter_primary_share_class(
    frame: pd.DataFrame,
    date_column: str = "Date",
    company_column: str = "PERMCO",
    mkt_cap_column: str = "MKT Cap",
) -> pd.DataFrame:
    """Keep only the largest-market-cap PERMNO per PERMCO for each date."""
    return (
        frame.sort_values(by=[date_column, company_column, mkt_cap_column], ascending=[True, True, False])
        .drop_duplicates(subset=[date_column, company_column], keep="first")
        .copy()
    )


def prepare_initial_stock_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the first end-to-end cleaning block from the legacy notebook."""
    cleaned = rename_and_parse_date(frame)
    cleaned = filter_supported_exchanges(cleaned)
    cleaned = filter_common_shares(cleaned)
    cleaned = clean_basic_columns(cleaned)
    cleaned = build_adjusted_return(cleaned)
    cleaned = add_market_cap_columns(cleaned)
    cleaned = filter_primary_share_class(cleaned)
    return cleaned

