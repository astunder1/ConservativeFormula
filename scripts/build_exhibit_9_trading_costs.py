from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.exhibit9 import build_trading_cost_summary, render_trading_cost_table

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

RETURNS_PATH = PROCESSED_DIR / "conservative_portfolio_returns.parquet"
SELECTIONS_PATH = PROCESSED_DIR / "portfolio_selections.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exhibit 9 trading cost summary.")
    parser.add_argument("--start-period", default="1929-01", help="Sample start in YYYY-MM.")
    parser.add_argument("--end-period", default="2016-12", help="Sample end in YYYY-MM.")
    parser.add_argument("--low-cost-bps", type=float, default=10.0)
    parser.add_argument("--high-cost-bps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conservative_returns = pd.read_parquet(RETURNS_PATH)
    selections = pd.read_parquet(SELECTIONS_PATH)

    summary, turnover = build_trading_cost_summary(
        conservative_returns=conservative_returns,
        selections=selections,
        start_period=pd.Period(args.start_period, freq="M"),
        end_period=pd.Period(args.end_period, freq="M"),
        low_cost_bps=args.low_cost_bps,
        high_cost_bps=args.high_cost_bps,
    )

    summary_path = OUTPUT_DIR / "exhibit_9_trading_costs.csv"
    turnover_path = OUTPUT_DIR / "exhibit_9_turnover_detail.csv"
    figure_path = OUTPUT_DIR / "exhibit_9_trading_costs.png"

    summary.to_csv(summary_path, index=False)
    turnover.to_csv(turnover_path, index=False)
    render_trading_cost_table(summary, figure_path)

    print(f"Saved Exhibit 9 summary table to {summary_path}")
    print(f"Saved Exhibit 9 turnover detail to {turnover_path}")
    print(f"Saved Exhibit 9 figure to {figure_path}")
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == "__main__":
    main()
