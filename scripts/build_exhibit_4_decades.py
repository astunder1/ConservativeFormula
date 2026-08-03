from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.data_loading import load_stock_data
from conservative_formula.exhibits import build_decade_summary, build_value_weighted_market_returns


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONSERVATIVE_INPUT = DEFAULT_INPUT_DIR / "conservative_portfolio_returns.parquet"
DEFAULT_SPECULATIVE_INPUT = DEFAULT_INPUT_DIR / "speculative_portfolio_returns.parquet"
DEFAULT_MARKET_INPUT = DEFAULT_INPUT_DIR / "crsp_us_monthly_initial.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_TABLE_OUTPUT = DEFAULT_OUTPUT_DIR / "exhibit_4_decades.csv"
DEFAULT_PLOT_OUTPUT = DEFAULT_OUTPUT_DIR / "exhibit_4_decades.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Exhibit 4 style decade-return table and chart."
    )
    parser.add_argument("--conservative-input", default=str(DEFAULT_CONSERVATIVE_INPUT))
    parser.add_argument("--speculative-input", default=str(DEFAULT_SPECULATIVE_INPUT))
    parser.add_argument("--market-input", default=str(DEFAULT_MARKET_INPUT))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE_OUTPUT))
    parser.add_argument("--plot-output", default=str(DEFAULT_PLOT_OUTPUT))
    parser.add_argument("--end-date", default="2016-12-31")
    return parser.parse_args()


def plot_decades(decade_averages, plot_output: Path) -> None:
    plt.figure(figsize=(12, 6))

    bar_width = 0.25
    x = range(len(decade_averages))

    plt.bar(
        [p - bar_width for p in x],
        decade_averages["Conservative"],
        width=bar_width,
        color="black",
        label="Conservative",
    )
    plt.bar(
        x,
        decade_averages["Market"],
        width=bar_width,
        color="gray",
        label="Market",
    )
    plt.bar(
        [p + bar_width for p in x],
        decade_averages["Speculative"],
        width=bar_width,
        color="lightgray",
        edgecolor="black",
        label="Speculative",
    )

    plt.xticks(x, labels=[f"{int(decade)}s" for decade in decade_averages["Decade"]])
    plt.ylabel("Average Annualized Return (%)", fontsize=14)
    plt.title("Never a Lost Decade", fontsize=16, fontweight="bold")
    plt.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    plt.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15), fontsize=14)
    plt.grid(axis="y", visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
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
    market_returns["YearMonth"] = market_returns["YearMonth"].dt.to_timestamp()
    decade_summary = build_decade_summary(conservative, speculative, market_returns)

    table_output.parent.mkdir(parents=True, exist_ok=True)
    decade_summary.to_csv(table_output, index=False)
    plot_decades(decade_summary, plot_output)

    print(f"Saved Exhibit 4 table to {table_output}")
    print(f"Saved Exhibit 4 plot to {plot_output}")
    print(decade_summary.to_string(index=False))


if __name__ == "__main__":
    main()
