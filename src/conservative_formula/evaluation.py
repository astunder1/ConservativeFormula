"""Performance evaluation helpers for the Conservative Formula project."""

import math

import pandas as pd


def cumulative_return(return_series: pd.Series) -> pd.Series:
    """Convert a simple return series into a cumulative return index."""
    return (1.0 + return_series.fillna(0.0)).cumprod()


def annualized_return(return_series: pd.Series, periods_per_year: int = 12) -> float:
    """Compute the annualized geometric return from a periodic return series."""
    clean = return_series.dropna()
    if clean.empty:
        return math.nan
    total_return = (1.0 + clean).prod()
    n_periods = len(clean)
    return total_return ** (periods_per_year / n_periods) - 1.0


def annualized_volatility(return_series: pd.Series, periods_per_year: int = 12) -> float:
    """Compute annualized volatility from a periodic return series."""
    clean = return_series.dropna()
    if clean.empty:
        return math.nan
    return clean.std() * math.sqrt(periods_per_year)


def sharpe_ratio_simple(return_series: pd.Series, periods_per_year: int = 12) -> float:
    """Compute a simple Sharpe ratio assuming a zero risk-free rate."""
    ann_return = annualized_return(return_series, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(return_series, periods_per_year=periods_per_year)
    if pd.isna(ann_return) or pd.isna(ann_vol) or ann_vol == 0:
        return math.nan
    return ann_return / ann_vol


def summarize_return_series(
    return_series: pd.Series,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """Build a compact performance summary for a periodic return series."""
    clean = return_series.dropna()
    cumulative = cumulative_return(clean)
    return {
        "n_periods": float(len(clean)),
        "total_return": float(cumulative.iloc[-1] - 1.0) if not cumulative.empty else math.nan,
        "annualized_return": float(annualized_return(clean, periods_per_year=periods_per_year)),
        "annualized_volatility": float(
            annualized_volatility(clean, periods_per_year=periods_per_year)
        ),
        "sharpe_simple": float(sharpe_ratio_simple(clean, periods_per_year=periods_per_year)),
    }
