import numpy as np
import pandas as pd
import pytest

from conservative_formula.factors import calculate_12_1_momentum


def _reference_momentum(returns: list[float]) -> list[float]:
    """Independent, non-vectorized reference: momentum(t) = prod(1+ret[t-11:t]) - 1,
    i.e. the compounded return over the 11 months ending one month before t."""
    n = len(returns)
    out = [np.nan] * n
    for t in range(n):
        if t < 11:
            continue
        window = returns[t - 11 : t]
        prod = 1.0
        for r in window:
            prod *= 1.0 + r
        out[t] = prod - 1.0
    return out


def _build_panel(returns: list[float], permno: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=len(returns), freq="ME")
    return pd.DataFrame({"PERMNO": permno, "Date": dates, "RET ADJ": returns})


def _momentum(frame: pd.DataFrame) -> pd.Series:
    result = calculate_12_1_momentum(
        frame, stock_column="PERMNO", date_column="Date", return_column="RET ADJ"
    )
    return result["Momentum"]


def test_momentum_matches_reference_implementation():
    returns = [0.01 * (i + 1) for i in range(14)]
    frame = _build_panel(returns)

    result = _momentum(frame)
    expected = _reference_momentum(returns)

    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True, rtol=1e-10)


def test_momentum_hand_computed_value():
    # 12 months of a flat 1% return. By hand: momentum at month 12 (index 11) is the
    # compounded return over the first 11 months, skipping the most recent month.
    returns = [0.01] * 12
    frame = _build_panel(returns)

    result = _momentum(frame)

    expected = 1.01**11 - 1.0
    assert result.iloc[11] == pytest.approx(expected, rel=1e-10)
    assert result.iloc[:11].isna().all()


def test_momentum_is_immune_to_a_price_split_when_returns_are_correct():
    """A 2:1-split-like price cut, with RET ADJ (CRSP-computed, already split-adjusted)
    left untouched, must not distort momentum. This is the exact failure mode of the old
    ALTPRC-ratio implementation: a raw price ratio across a split date produced a spurious
    ~-50% momentum reading even though the true return was unaffected."""
    returns = [0.02] * 20
    frame = _build_panel(returns)
    frame["ALTPRC"] = 100.0
    frame.loc[frame.index[10:], "ALTPRC"] = 50.0  # simulated 2:1 split, RET ADJ unchanged

    result = _momentum(frame)
    expected = _reference_momentum(returns)

    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True, rtol=1e-10)


def test_momentum_does_not_leak_across_stocks():
    returns_a = [0.01] * 14
    returns_b = [0.05] * 14
    frame = pd.concat(
        [_build_panel(returns_a, permno=1), _build_panel(returns_b, permno=2)],
        ignore_index=True,
    )

    result = calculate_12_1_momentum(
        frame, stock_column="PERMNO", date_column="Date", return_column="RET ADJ"
    )

    momentum_a = result.loc[result["PERMNO"] == 1, "Momentum"].reset_index(drop=True)
    momentum_b = result.loc[result["PERMNO"] == 2, "Momentum"].reset_index(drop=True)

    np.testing.assert_allclose(
        momentum_a.to_numpy(), _reference_momentum(returns_a), equal_nan=True, rtol=1e-10
    )
    np.testing.assert_allclose(
        momentum_b.to_numpy(), _reference_momentum(returns_b), equal_nan=True, rtol=1e-10
    )
