from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.data_loading import load_stock_data
from conservative_formula.factors import (
    add_dividend_yield,
    add_net_payout_yield,
    calculate_12_1_momentum,
    calculate_rolling_volatility,
    drop_duplicate_stock_dates,
    keep_stocks_with_min_history,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "crsp_us_monthly_initial.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "crsp_us_monthly_factors.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the factor-ready monthly panel for the Conservative Formula replication."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def summarize(frame) -> None:
    print(f"Rows: {len(frame):,}")
    print(f"Unique PERMNOs: {frame['PERMNO'].nunique():,}")
    print(f"Date range: {frame['Date'].min()} to {frame['Date'].max()}")
    print("Missing-rate snapshot:")
    print(
        frame[
            ["Div Yield", "Net_Payout_Yield", "Momentum", "Volatility"]
        ]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading initial panel from {input_path}")
    frame = load_stock_data(input_path)

    print("Applying factor construction pipeline...")
    frame = keep_stocks_with_min_history(frame, stock_column="PERMNO", date_column="Date", min_periods=36)
    frame = add_dividend_yield(frame, dividend_column="DIVAMT", price_column="ALTPRC", output_column="Div Yield")
    frame = drop_duplicate_stock_dates(
        frame,
        stock_column="PERMNO",
        date_column="Date",
        tie_breaker_column="DIVAMT",
    )
    frame = add_net_payout_yield(
        frame,
        stock_column="PERMNO",
        shares_column="SHROUT",
        div_yield_column="Div Yield",
        average_window=24,
    )
    frame = calculate_12_1_momentum(
        frame,
        stock_column="PERMNO",
        date_column="Date",
        price_column="ALTPRC",
        momentum_column="Momentum",
    )
    frame = calculate_rolling_volatility(
        frame,
        stock_column="PERMNO",
        date_column="Date",
        return_column="RET ADJ",
        volatility_column="Volatility",
        window=36,
    )
    frame = frame.dropna(subset=["Momentum", "Volatility"]).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    print(f"Saved factor panel to {output_path}")
    summarize(frame)


if __name__ == "__main__":
    main()
