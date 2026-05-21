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
    dividend_column: str = "DIVAMT",
    price_column: str = "ALTPRC",
    output_column: str = "Div Yield",
) -> pd.DataFrame:
    """Add the one-period dividend yield used in the legacy notebook."""
    enriched = frame.copy()
    enriched[output_column] = enriched[dividend_column] / enriched[price_column]
    enriched[output_column] = enriched[output_column].fillna(0)
    return enriched


# def add_trailing_dividend_yield(
#     frame: pd.DataFrame,
#     stock_column: str = "PERMNO",
#     dividend_yield_column: str = "Div Yield",
#     output_column: str = "1-yr Div Yield",
#     window: int = 24,
# ) -> pd.DataFrame:
#     """Add a legacy notebook feature that is not in the paper's core NPY definition."""
#     enriched = frame.copy()
#     enriched[output_column] = (
#         enriched.groupby(stock_column)[dividend_yield_column]
#         .rolling(window=window, min_periods=window)
#         .sum()
#         .reset_index(level=0, drop=True)
#     )
#     return enriched


# def drop_duplicate_stock_dates(
#     frame: pd.DataFrame,
#     stock_column: str = "PERMNO",
#     date_column: str = "Date",
#     tie_breaker_column: str = "DIVAMT",
# ) -> pd.DataFrame:
#     """Resolve duplicate stock-date rows the same way as the legacy notebook."""
#     deduped = frame.sort_values(
#         [stock_column, date_column, tie_breaker_column],
#         ascending=[True, True, False],
#     )
#     return deduped.drop_duplicates(subset=[stock_column, date_column], keep="first").copy()


def add_net_payout_yield(
    frame: pd.DataFrame,
    stock_column: str = "PERMNO",
    shares_column: str = "SHROUT",
    div_yield_column: str = "Div Yield",
    average_window: int = 24,
) -> pd.DataFrame:
    """Add net payout yield and components required to calculate it."""
    enriched = frame.copy()
    enriched = enriched.sort_values([stock_column, "Date"])
    enriched["Shares_24M_Avg"] = (
        enriched.groupby(stock_column)[shares_column]
        .rolling(window=average_window, min_periods=average_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    enriched["Net_Shares_Change"] = (enriched["Shares_24M_Avg"] / enriched[shares_column]) - 1
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
    price_column: str,
    momentum_column: str = "Momentum",
) -> pd.DataFrame:
    """Reproduce the legacy notebook's 12-1 momentum definition from prices."""
    enriched = frame.sort_values(by=[stock_column, date_column]).copy()
    grouped_price = enriched.groupby(stock_column)[price_column]
    enriched[momentum_column] = grouped_price.shift(1) / grouped_price.shift(12) - 1
    return enriched