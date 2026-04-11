from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.data_loading import load_stock_data
from conservative_formula.portfolio import (
    build_equal_weight_returns,
    get_rebalance_dates,
    run_conservative_selection,
    run_speculative_selection,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "crsp_us_monthly_factors.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONSERVATIVE_OUTPUT = DEFAULT_OUTPUT_DIR / "conservative_portfolio_returns.parquet"
DEFAULT_SPECULATIVE_OUTPUT = DEFAULT_OUTPUT_DIR / "speculative_portfolio_returns.parquet"
DEFAULT_SELECTIONS_OUTPUT = DEFAULT_OUTPUT_DIR / "portfolio_selections.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conservative and speculative portfolio return series."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--conservative-output", default=str(DEFAULT_CONSERVATIVE_OUTPUT))
    parser.add_argument("--speculative-output", default=str(DEFAULT_SPECULATIVE_OUTPUT))
    parser.add_argument("--selections-output", default=str(DEFAULT_SELECTIONS_OUTPUT))
    parser.add_argument("--start-date", default="1929-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    return parser.parse_args()


def attach_period_label(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.reset_index().copy()
    labeled["YearMonth"] = labeled["Date"].dt.to_period("M").astype(str)
    return labeled


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    conservative_output = Path(args.conservative_output)
    speculative_output = Path(args.speculative_output)
    selections_output = Path(args.selections_output)

    print(f"Loading factor panel from {input_path}")
    frame = load_stock_data(input_path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame[
        (frame["Date"] >= pd.Timestamp(args.start_date))
        & (frame["Date"] <= pd.Timestamp(args.end_date))
    ].copy()

    rebalance_dates = get_rebalance_dates(frame, date_column="Date")
    conservative_returns = []
    speculative_returns = []
    selections = []

    print("Building quarterly portfolios...")
    for i in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates.iloc[i]
        next_rebalance_date = rebalance_dates.iloc[i + 1]
        rebalance_frame = frame.loc[frame["Date"] == rebalance_date]

        conservative_selection = run_conservative_selection(rebalance_frame)
        conservative_selection["Portfolio"] = "Conservative"
        conservative_selection["RebalanceDate"] = rebalance_date
        selections.append(conservative_selection)

        speculative_selection = run_speculative_selection(rebalance_frame)
        speculative_selection["Portfolio"] = "Speculative"
        speculative_selection["RebalanceDate"] = rebalance_date
        selections.append(speculative_selection)

        conservative_returns.append(
            build_equal_weight_returns(
                frame,
                selected_ids=conservative_selection["PERMNO"].values,
                start_date=rebalance_date,
                end_date=next_rebalance_date,
                id_column="PERMNO",
                date_column="Date",
                return_column="RET ADJ",
            )
        )
        speculative_returns.append(
            build_equal_weight_returns(
                frame,
                selected_ids=speculative_selection["PERMNO"].values,
                start_date=rebalance_date,
                end_date=next_rebalance_date,
                id_column="PERMNO",
                date_column="Date",
                return_column="RET ADJ",
            )
        )

    conservative_df = pd.concat(conservative_returns).sort_index()
    speculative_df = pd.concat(speculative_returns).sort_index()
    selections_df = pd.concat(selections, ignore_index=True)

    conservative_df = attach_period_label(conservative_df)
    speculative_df = attach_period_label(speculative_df)

    conservative_output.parent.mkdir(parents=True, exist_ok=True)
    conservative_df.to_parquet(conservative_output, index=False)
    speculative_df.to_parquet(speculative_output, index=False)
    selections_df.to_parquet(selections_output, index=False)

    print(f"Saved conservative returns to {conservative_output}")
    print(f"Saved speculative returns to {speculative_output}")
    print(f"Saved portfolio selections to {selections_output}")
    print(f"Conservative months: {len(conservative_df):,}")
    print(f"Speculative months: {len(speculative_df):,}")
    print(f"Selection rows: {len(selections_df):,}")


if __name__ == "__main__":
    main()


