import numpy as np
import pandas as pd
import pytest

from conservative_formula.cleaning import build_adjusted_return, filter_supported_exchanges


def _base_frame(**overrides) -> pd.DataFrame:
    row = {
        "PERMNO": 1,
        "Date": pd.Timestamp("2000-01-31"),
        "RET": 0.0,
        "DLRET": np.nan,
        "DLSTCD": np.nan,
        "ALTPRC": 10.0,
        "SHROUT": 100.0,
        "EXCHCD": "NYSE",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_delisting_return_compounds_with_ordinary_return():
    frame = _base_frame(RET=-0.3125, DLRET=0.2727, DLSTCD=560)
    result = build_adjusted_return(frame)

    expected = (1.0 - 0.3125) * (1.0 + 0.2727) - 1.0
    assert result["RET ADJ"].iloc[0] == pytest.approx(expected)


def test_delisting_return_falls_back_to_dlret_alone_when_ret_missing():
    frame = _base_frame(RET=np.nan, DLRET=-0.2, DLSTCD=560)
    result = build_adjusted_return(frame)

    assert result["RET ADJ"].iloc[0] == pytest.approx(-0.2)


def test_no_delisting_uses_ordinary_return():
    frame = _base_frame(RET=0.05)
    result = build_adjusted_return(frame)

    assert result["RET ADJ"].iloc[0] == pytest.approx(0.05)


def test_performance_delisting_missing_dlret_imputes_by_exchange():
    nasdaq = _base_frame(RET=np.nan, DLRET=np.nan, DLSTCD=560, EXCHCD="NASDAQ")
    nyse = _base_frame(RET=np.nan, DLRET=np.nan, DLSTCD=560, EXCHCD="NYSE")

    result_nasdaq = build_adjusted_return(nasdaq)
    result_nyse = build_adjusted_return(nyse)

    assert result_nasdaq["RET ADJ"].iloc[0] == pytest.approx(-0.55)
    assert result_nyse["RET ADJ"].iloc[0] == pytest.approx(-0.3)


def test_terminal_delisting_row_survives_missing_exchcd_via_ffill():
    frame = pd.DataFrame(
        {
            "PERMNO": [1, 1, 1],
            "Date": pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"]),
            "EXCHCD": [1, 1, np.nan],
        }
    )
    result = filter_supported_exchanges(frame)

    assert len(result) == 3
    assert (result["EXCHCD"] == "NYSE").all()


def test_stock_never_listed_on_a_supported_exchange_is_dropped():
    frame = pd.DataFrame(
        {
            "PERMNO": [1, 1],
            "Date": pd.to_datetime(["2000-01-31", "2000-02-29"]),
            "EXCHCD": [np.nan, np.nan],
        }
    )
    result = filter_supported_exchanges(frame)

    assert result.empty
