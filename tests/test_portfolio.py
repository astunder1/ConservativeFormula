import pandas as pd

from conservative_formula.cleaning import filter_primary_share_class
from conservative_formula.portfolio import get_rebalance_dates, run_conservative_selection, run_speculative_selection


def _synthetic_universe(n_stocks: int, date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": range(1, n_stocks + 1),
            "Date": date,
            "MKT Cap": [n_stocks - i for i in range(n_stocks)],
            "Volatility": [0.10 + 0.001 * i for i in range(n_stocks)],
            "Momentum": [0.20 - 0.001 * i for i in range(n_stocks)],
            "Net_Payout_Yield": [0.05 - 0.0005 * i for i in range(n_stocks)],
        }
    )


def test_conservative_and_speculative_selection_pick_100_stocks():
    date = pd.Timestamp("2000-03-31")
    frame = _synthetic_universe(2000, date)

    conservative = run_conservative_selection(frame)
    speculative = run_speculative_selection(frame)

    assert len(conservative) == 100
    assert len(speculative) == 100
    assert conservative["PERMNO"].is_unique
    assert speculative["PERMNO"].is_unique


def test_conservative_and_speculative_selections_do_not_overlap():
    date = pd.Timestamp("2000-03-31")
    frame = _synthetic_universe(2000, date)

    conservative = set(run_conservative_selection(frame)["PERMNO"])
    speculative = set(run_speculative_selection(frame)["PERMNO"])

    assert conservative.isdisjoint(speculative)


def test_get_rebalance_dates_keeps_only_quarter_ends():
    dates = pd.date_range("2000-01-31", "2000-12-31", freq="ME")
    frame = pd.DataFrame({"Date": dates})

    rebalance_dates = get_rebalance_dates(frame)

    assert list(rebalance_dates.dt.month) == [3, 6, 9, 12]
    assert rebalance_dates.is_monotonic_increasing


def test_filter_primary_share_class_keeps_larger_market_cap_permno():
    frame = pd.DataFrame(
        {
            "PERMNO": [10, 11, 20],
            "PERMCO": [1, 1, 2],
            "Date": [pd.Timestamp("2000-01-31")] * 3,
            "MKT Cap": [500.0, 800.0, 300.0],
        }
    )

    result = filter_primary_share_class(frame)

    assert set(result["PERMNO"]) == {11, 20}
