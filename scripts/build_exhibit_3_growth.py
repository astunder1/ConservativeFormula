from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.data_loading import load_stock_data
from conservative_formula.exhibits import align_growth_series, build_value_weighted_market_returns


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONSERVATIVE_INPUT = DEFAULT_INPUT_DIR / "conservative_portfolio_returns.parquet"
DEFAULT_SPECULATIVE_INPUT = DEFAULT_INPUT_DIR / "speculative_portfolio_returns.parquet"
DEFAULT_MARKET_INPUT = DEFAULT_INPUT_DIR / "crsp_us_monthly_initial.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_TABLE_OUTPUT = DEFAULT_OUTPUT_DIR / "exhibit_3_growth.csv"
DEFAULT_PLOT_OUTPUT = DEFAULT_OUTPUT_DIR / "exhibit_3_growth.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Exhibit 3 style dollar-growth table and chart."
    )
    parser.add_argument("--conservative-input", default=str(DEFAULT_CONSERVATIVE_INPUT))
    parser.add_argument("--speculative-input", default=str(DEFAULT_SPECULATIVE_INPUT))
    parser.add_argument("--market-input", default=str(DEFAULT_MARKET_INPUT))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE_OUTPUT))
    parser.add_argument("--plot-output", default=str(DEFAULT_PLOT_OUTPUT))
    parser.add_argument("--end-date", default="2016-12-31")
    return parser.parse_args()


def plot_growth(exhibit: pd.DataFrame, plot_output: Path) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(
        exhibit["YearMonth"],
        exhibit["Conservative"],
        label="Conservative",
        color="black",
        linewidth=2,
    )
    plt.plot(
        exhibit["YearMonth"],
        exhibit["Market"],
        label="Market",
        color="grey",
        linestyle="-.",
        linewidth=2,
    )
    plt.plot(
        exhibit["YearMonth"],
        exhibit["Speculative"],
        label="Speculative",
        color="darkgrey",
        linestyle="--",
        linewidth=2,
    )

    plt.yscale("log")
    plt.yticks(
        [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
        ["$1", "$10", "$100", "$1,000", "$10,000", "$100,000", "$1,000,000", "$10,000,000", "$100,000,000"],
        fontsize=12,
    )
    plt.minorticks_off()
    plt.title("Value of $100 Invested in 1929", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.grid(False)
    plt.grid(axis="y", visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fancybox=True,
        shadow=False,
        ncol=3,
        fontsize=14,
        frameon=False,
    )
    plt.tight_layout()
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_output, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    conservative_input = Path(args.conservative_input)
    speculative_input = Path(args.speculative_input)
    market_input = Path(args.market_input)
    table_output = Path(args.table_output)
    plot_output = Path(args.plot_output)

    end_date = pd.Timestamp(args.end_date)

    print(f"Loading conservative returns from {conservative_input}")
    conservative = load_stock_data(conservative_input)
    conservative = conservative[pd.to_datetime(conservative["Date"]) <= end_date].copy()
    print(f"Loading speculative returns from {speculative_input}")
    speculative = load_stock_data(speculative_input)
    speculative = speculative[pd.to_datetime(speculative["Date"]) <= end_date].copy()
    print(f"Loading cleaned market panel from {market_input}")
    market_panel = load_stock_data(market_input)
    market_panel = market_panel[pd.to_datetime(market_panel["Date"]) <= end_date].copy()

    market_returns = build_value_weighted_market_returns(market_panel)
    exhibit = align_growth_series(conservative, speculative, market_returns)

    table_output.parent.mkdir(parents=True, exist_ok=True)
    exhibit.to_csv(table_output, index=False)
    plot_growth(exhibit, plot_output)

    print(f"Saved Exhibit 3 table to {table_output}")
    print(f"Saved Exhibit 3 plot to {plot_output}")
    print(exhibit.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
