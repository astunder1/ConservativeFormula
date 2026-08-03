"""Factor construction helpers for the Conservative Formula project."""

import pandas as pd


def keep_stocks_with_min_history(
    frame: pd.DataFrame,
    stock_column: str = "PERMNO",
    return_column: str = "RET ADJ",
    min_periods: int = 36,
) -> pd.DataFrame:
    """Keep stocks with at least the required number of usable monthly return observations."""
    cleaned = frame.copy()
    cleaned[return_column] = pd.to_numeric(cleaned[return_column], errors="coerce")
    valid_counts = cleaned.groupby(stock_column)[return_column].count()
    valid_ids = valid_counts[valid_counts >= min_periods].index
    return cleaned.loc[cleaned[stock_column].isin(valid_ids)].copy()


def add_dividend_yield(
    frame: pd.DataFrame,
    stock_column: str = "PERMNO",
    date_column: str = "Date",
    dividend_column: str = "DIVAMT",
    price_column: str = "ALTPRC",
    output_column: str = "Div Yield",
    window: int = 12,
) -> pd.DataFrame:
    """Add a trailing 12-month dividend yield."""
    enriched = frame.sort_values([stock_column, date_column]).copy()
    trailing_dividends = (
        enriched.groupby(stock_column)[dividend_column]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    enriched[output_column] = trailing_dividends / enriched[price_column]
    return enriched


def add_net_payout_yield(
    frame: pd.DataFrame,
    stock_column: str = "PERMNO",
    shares_column: str = "SHROUT",
    shares_adj_factor_column: str = "CFACSHR",
    div_yield_column: str = "Div Yield",
    average_window: int = 24,
) -> pd.DataFrame:
    """Add net payout yield and components required to calculate it."""
    enriched = frame.copy()
    enriched = enriched.sort_values([stock_column, "Date"])
    adj_factor = enriched[shares_adj_factor_column].where(enriched[shares_adj_factor_column] > 0)
    enriched["Shares_Adj"] = enriched[shares_column] * adj_factor
    enriched["Shares_24M_Avg"] = (
        enriched.groupby(stock_column)["Shares_Adj"]
        .rolling(window=average_window, min_periods=average_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    enriched["Net_Shares_Change"] = (enriched["Shares_24M_Avg"] / enriched["Shares_Adj"]) - 1
    enriched["Net_Payout_Yield"] = enriched[div_yield_column] + enriched["Net_Shares_Change"]
    return enriched


def calculate_rolling_volatility(
    frame: pd.DataFrame,
    stock_column: str,
    date_column: str,
    return_column: str,
    volatility_column: str = "Volatility",
    window: int = 36,
) -> pd.DataFrame:
    """Calculate rolling return volatility for each stock."""
    enriched = frame.sort_values(by=[stock_column, date_column], ascending=[True, True]).copy()
    enriched[volatility_column] = (
        enriched.groupby(stock_column)[return_column]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )
    return enriched


def calculate_12_1_momentum(
    frame: pd.DataFrame,
    stock_column: str,
    date_column: str,
    return_column: str,
    momentum_column: str = "Momentum",
) -> pd.DataFrame:
    """Reproduce the legacy notebook's 12-1 momentum definition from returns."""
    enriched = frame.sort_values(by=[stock_column, date_column]).copy()
    gross = 1.0 + enriched[return_column]
    cum11 = gross.groupby(enriched[stock_column]).transform(lambda s: s.rolling(11).apply(lambda w: w.prod(), raw=True))
    enriched[momentum_column] = cum11.groupby(enriched[stock_column]).shift(1) - 1
    return enriched