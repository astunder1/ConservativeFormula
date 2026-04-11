"""Helpers for rebuilding presentation exhibits from processed pipeline outputs."""

from __future__ import annotations

import pandas as pd


def cumulate_returns(returns: pd.Series) -> pd.Series:
    """Cumulates simple returns into a growth index."""
    return (1.0 + returns).cumprod()


def build_value_weighted_market_returns(
    frame: pd.DataFrame,
    date_column: str = "Date",
    lagged_mkt_cap_column: str = "Lagged MKT Cap",
    return_column: str = "RET ADJ",
) -> pd.DataFrame:
    """Build a value-weighted market return series from the cleaned stock panel."""
    market = frame.copy()
    market["YearMonth"] = pd.to_datetime(market[date_column]).dt.to_period("M")
    market["Total_Prev_MKT_Cap"] = market.groupby("YearMonth")[lagged_mkt_cap_column].transform("sum")
    market["Weight"] = market[lagged_mkt_cap_column] / market["Total_Prev_MKT_Cap"]
    market["Weighted_Return"] = market[return_column] * market["Weight"]

    value_weighted_returns = (
        market.groupby("YearMonth")["Weighted_Return"]
        .sum()
        .reset_index()
        .rename(columns={"Weighted_Return": "Portfolio Returns"})
    )
    return value_weighted_returns.iloc[1:].copy()


def build_growth_index(
    returns_frame: pd.DataFrame,
    return_column: str = "Portfolio Returns",
    base_value: float = 100.0,
) -> pd.Series:
    """Convert periodic returns into a rebased cumulative wealth index."""
    cumulative = cumulate_returns(pd.to_numeric(returns_frame[return_column], errors="coerce"))
    if cumulative.empty:
        return cumulative
    return cumulative / cumulative.iloc[0] * base_value


def align_growth_series(
    conservative_returns: pd.DataFrame,
    speculative_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Align the three Exhibit 3 growth series on one monthly timeline."""
    conservative = conservative_returns.copy()
    speculative = speculative_returns.copy()
    market = market_returns.copy()

    conservative["YearMonth"] = pd.to_datetime(conservative["YearMonth"])
    speculative["YearMonth"] = pd.to_datetime(speculative["YearMonth"])
    market["YearMonth"] = market["YearMonth"].dt.to_timestamp()

    conservative_aligned = conservative[["YearMonth", "Portfolio Returns"]].rename(
        columns={"Portfolio Returns": "ConservativeReturn"}
    )
    speculative_aligned = speculative[["YearMonth", "Portfolio Returns"]].rename(
        columns={"Portfolio Returns": "SpeculativeReturn"}
    )
    market_aligned = market[["YearMonth", "Portfolio Returns"]].rename(
        columns={"Portfolio Returns": "MarketReturn"}
    )

    aligned = (
        conservative_aligned.merge(speculative_aligned, on="YearMonth", how="inner")
        .merge(market_aligned, on="YearMonth", how="inner")
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )

    exhibit = pd.DataFrame({"YearMonth": aligned["YearMonth"]})
    exhibit["Conservative"] = build_growth_index(
        aligned.rename(columns={"ConservativeReturn": "Portfolio Returns"})
    ).reset_index(drop=True)
    exhibit["Speculative"] = build_growth_index(
        aligned.rename(columns={"SpeculativeReturn": "Portfolio Returns"})
    ).reset_index(drop=True)
    exhibit["Market"] = build_growth_index(
        aligned.rename(columns={"MarketReturn": "Portfolio Returns"})
    ).reset_index(drop=True)
    return exhibit


def calculate_annualized_decade_averages(
    portfolio_returns: pd.DataFrame,
    year_column: str = "YearMonth",
    return_column: str = "Portfolio Returns",
    periods_per_year: int = 12,
) -> pd.Series:
    """Calculate annualized geometric returns grouped by decade."""
    frame = portfolio_returns.copy()
    frame[year_column] = pd.to_datetime(frame[year_column])
    frame["Decade"] = (frame[year_column].dt.year // 10) * 10
    frame = frame.loc[frame["Decade"] >= 1930].copy()

    return frame.groupby("Decade")[return_column].apply(
        lambda group: (1.0 + group).prod() ** (1.0 / (len(group) / periods_per_year)) - 1.0
    )


def build_decade_summary(
    conservative_returns: pd.DataFrame,
    speculative_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Exhibit 4 decade-level annualized return summary."""
    decade_averages_cons = calculate_annualized_decade_averages(conservative_returns)
    decade_averages_spec = calculate_annualized_decade_averages(speculative_returns)
    decade_averages_market = calculate_annualized_decade_averages(market_returns)

    return pd.DataFrame(
        {
            "Conservative": decade_averages_cons,
            "Speculative": decade_averages_spec,
            "Market": decade_averages_market,
        }
    ).reset_index()
