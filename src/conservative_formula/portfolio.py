"""Portfolio construction helpers for the Conservative Formula project."""

import pandas as pd


QUARTER_END_MONTHS = (3, 6, 9, 12)


def filter_to_quarter_end(frame: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """Keep only quarter-end months used for rebalancing."""
    return frame.loc[frame[date_column].dt.month.isin(QUARTER_END_MONTHS)].copy()


def filter_top_n_by_market_cap(
    frame: pd.DataFrame,
    date_column: str = "Date",
    mkt_cap_column: str = "MKT Cap",
    top_n: int = 1000,
) -> pd.DataFrame:
    """Select the largest stocks by market cap for each rebalance date."""
    return (
        frame.sort_values(by=[date_column, mkt_cap_column], ascending=[True, False])
        .groupby(date_column, group_keys=False)
        .head(top_n)
        .copy()
    )


def split_by_volatility(
    frame: pd.DataFrame,
    date_column: str = "Date",
    volatility_column: str = "Volatility",
    group_size: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each date into low-volatility and high-volatility groups."""
    ranked = frame.copy()
    ranked["Volatility Rank"] = ranked.groupby(date_column)[volatility_column].rank(
        method="first",
        ascending=True,
    )

    low_volatility_list = []
    high_volatility_list = []

    for _, group in ranked.groupby(date_column):
        total_stocks = len(group)
        dynamic_group_size = total_stocks // 2 if total_stocks < group_size * 2 else group_size
        low_volatility_list.append(group.loc[group["Volatility Rank"] <= dynamic_group_size].copy())
        high_volatility_list.append(group.loc[group["Volatility Rank"] > dynamic_group_size].copy())

    low_volatility = pd.concat(low_volatility_list, ignore_index=True)
    high_volatility = pd.concat(high_volatility_list, ignore_index=True)
    return low_volatility, high_volatility


def compute_combined_rank(
    frame: pd.DataFrame,
    date_column: str = "Date",
    momentum_column: str = "Momentum",
    npy_column: str = "Net_Payout_Yield",
    ascending: bool = False,
) -> pd.DataFrame:
    """Average momentum and NPY ranks within each date."""
    ranked = frame.copy()
    ranked["Momentum_Rank"] = ranked.groupby(date_column)[momentum_column].rank(
        method="first",
        ascending=ascending,
    )
    ranked["NPY_Rank"] = ranked.groupby(date_column)[npy_column].rank(
        method="first",
        ascending=ascending,
    )
    ranked["Combined_Rank"] = (ranked["Momentum_Rank"] + ranked["NPY_Rank"]) / 2
    return ranked


def select_top_stocks_by_rank(
    frame: pd.DataFrame,
    date_column: str = "Date",
    rank_column: str = "Combined_Rank",
    top_n: int = 100,
) -> pd.DataFrame:
    """Select the strongest ranked names for each date."""
    return (
        frame.sort_values(by=[date_column, rank_column], ascending=[True, True])
        .groupby(date_column, group_keys=False)
        .head(top_n)
        .copy()
    )


def get_rebalance_dates(frame: pd.DataFrame, date_column: str = "Date") -> pd.Series:
    """Return sorted quarter-end rebalance dates present in the data."""
    quarter_end = filter_to_quarter_end(frame, date_column=date_column)
    return quarter_end[date_column].drop_duplicates().sort_values()


def build_equal_weight_returns(
    frame: pd.DataFrame,
    selected_ids: pd.Series | list,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    id_column: str = "PERMNO",
    date_column: str = "Date",
    return_column: str = "RET ADJ",
) -> pd.DataFrame:
    """Compute monthly equal-weight returns over the post-rebalance holding window."""
    holding_period_mask = (frame[date_column] > start_date) & (frame[date_column] <= end_date)
    holding_period_data = frame.loc[holding_period_mask & frame[id_column].isin(selected_ids)].copy()
    returns_pivot = holding_period_data.pivot(index=date_column, columns=id_column, values=return_column)
    portfolio_return = returns_pivot.mean(axis=1).to_frame(name="Portfolio Returns")
    return portfolio_return


def run_conservative_selection(
    rebalance_frame: pd.DataFrame,
    date_column: str = "Date",
    mkt_cap_column: str = "MKT Cap",
    volatility_column: str = "Volatility",
    momentum_column: str = "Momentum",
    npy_column: str = "Net_Payout_Yield",
    top_universe_n: int = 1000,
    low_vol_group_size: int = 500,
    top_portfolio_n: int = 100,
) -> pd.DataFrame:
    """Run the paper-faithful conservative selection on a rebalance-date cross section."""
    top_universe = filter_top_n_by_market_cap(
        rebalance_frame,
        date_column=date_column,
        mkt_cap_column=mkt_cap_column,
        top_n=top_universe_n,
    )
    low_volatility, _ = split_by_volatility(
        top_universe,
        date_column=date_column,
        volatility_column=volatility_column,
        group_size=low_vol_group_size,
    )
    ranked = compute_combined_rank(
        low_volatility,
        date_column=date_column,
        momentum_column=momentum_column,
        npy_column=npy_column,
        ascending=False,
    )
    return select_top_stocks_by_rank(
        ranked,
        date_column=date_column,
        rank_column="Combined_Rank",
        top_n=top_portfolio_n,
    )


def run_speculative_selection(
    rebalance_frame: pd.DataFrame,
    date_column: str = "Date",
    mkt_cap_column: str = "MKT Cap",
    volatility_column: str = "Volatility",
    momentum_column: str = "Momentum",
    npy_column: str = "Net_Payout_Yield",
    top_universe_n: int = 1000,
    high_vol_group_size: int = 500,
    top_portfolio_n: int = 100,
) -> pd.DataFrame:
    """Run the opposite speculative selection on a rebalance-date cross section."""
    top_universe = filter_top_n_by_market_cap(
        rebalance_frame,
        date_column=date_column,
        mkt_cap_column=mkt_cap_column,
        top_n=top_universe_n,
    )
    _, high_volatility = split_by_volatility(
        top_universe,
        date_column=date_column,
        volatility_column=volatility_column,
        group_size=high_vol_group_size,
    )
    ranked = compute_combined_rank(
        high_volatility,
        date_column=date_column,
        momentum_column=momentum_column,
        npy_column=npy_column,
        ascending=True,
    )
    return select_top_stocks_by_rank(
        ranked,
        date_column=date_column,
        rank_column="Combined_Rank",
        top_n=top_portfolio_n,
    )


def equal_weight_portfolio(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Assign equal weights across all rows in a frame."""
    portfolio = frame.copy()
    count = len(portfolio.index)
    portfolio["weight"] = 0.0 if count == 0 else 1.0 / count
    return portfolio[[id_column, "weight"]]
