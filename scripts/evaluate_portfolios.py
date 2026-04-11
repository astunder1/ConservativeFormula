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
from conservative_formula.evaluation import cumulative_return, summarize_return_series


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONSERVATIVE_INPUT = DEFAULT_INPUT_DIR / "conservative_portfolio_returns.parquet"
DEFAULT_SPECULATIVE_INPUT = DEFAULT_INPUT_DIR / "speculative_portfolio_returns.parquet"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_INPUT_DIR / "portfolio_summary.csv"
DEFAULT_CUMULATIVE_OUTPUT = DEFAULT_INPUT_DIR / "portfolio_cumulative_returns.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Conservative Formula portfolio return series."
    )
    parser.add_argument("--conservative-input", default=str(DEFAULT_CONSERVATIVE_INPUT))
    parser.add_argument("--speculative-input", default=str(DEFAULT_SPECULATIVE_INPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--cumulative-output", default=str(DEFAULT_CUMULATIVE_OUTPUT))
    return parser.parse_args()


def extract_return_series(frame: pd.DataFrame) -> pd.Series:
    series = frame["Portfolio Returns"].copy()
    return series.astype(float)


def main() -> None:
    args = parse_args()
    conservative_input = Path(args.conservative_input)
    speculative_input = Path(args.speculative_input)
    summary_output = Path(args.summary_output)
    cumulative_output = Path(args.cumulative_output)

    print(f"Loading conservative returns from {conservative_input}")
    conservative = load_stock_data(conservative_input)
    print(f"Loading speculative returns from {speculative_input}")
    speculative = load_stock_data(speculative_input)

    conservative_series = extract_return_series(conservative)
    speculative_series = extract_return_series(speculative)
    spread_series = conservative_series.reset_index(drop=True) - speculative_series.reset_index(drop=True)

    summary_rows = []
    for label, series in [
        ("Conservative", conservative_series),
        ("Speculative", speculative_series),
        ("ConservativeMinusSpeculative", spread_series),
    ]:
        row = summarize_return_series(series, periods_per_year=12)
        row["portfolio"] = label
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)[
        [
            "portfolio",
            "n_periods",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_simple",
        ]
    ]

    cumulative_df = pd.DataFrame(
        {
            "YearMonth": conservative["YearMonth"].reset_index(drop=True),
            "Conservative": cumulative_return(conservative_series.reset_index(drop=True)),
            "Speculative": cumulative_return(speculative_series.reset_index(drop=True)),
            "ConservativeMinusSpeculativeSpread": cumulative_return(spread_series),
        }
    )

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)
    cumulative_df.to_parquet(cumulative_output, index=False)

    print(f"Saved summary table to {summary_output}")
    print(f"Saved cumulative return panel to {cumulative_output}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
