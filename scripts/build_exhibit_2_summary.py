from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.data_loading import load_stock_data
from conservative_formula.evaluation import annualized_return, annualized_volatility

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONSERVATIVE_INPUT = DEFAULT_INPUT_DIR / "conservative_portfolio_returns.parquet"
DEFAULT_SPECULATIVE_INPUT = DEFAULT_INPUT_DIR / "speculative_portfolio_returns.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_TABLE_OUTPUT = DEFAULT_OUTPUT_DIR / "exhibit_2_summary.csv"
DEFAULT_PLOT_OUTPUT = DEFAULT_OUTPUT_DIR / "exhibit_2_summary.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Exhibit 2 style U.S. return/volatility summary table and chart."
    )
    parser.add_argument("--conservative-input", default=str(DEFAULT_CONSERVATIVE_INPUT))
    parser.add_argument("--speculative-input", default=str(DEFAULT_SPECULATIVE_INPUT))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE_OUTPUT))
    parser.add_argument("--plot-output", default=str(DEFAULT_PLOT_OUTPUT))
    parser.add_argument("--end-date", default="2016-12-31")
    return parser.parse_args()


def plot_summary(summary: pd.DataFrame, plot_output: Path) -> None:
    colors = {"Conservative": "#2a78d6", "Speculative": "#eb6834"}

    plt.figure(figsize=(8, 5))
    for _, row in summary.iterrows():
        plt.scatter(
            row["Volatility (%)"],
            row["Return (compounded) (%)"],
            color=colors[row["Portfolio"]],
            label=row["Portfolio"],
            s=100,
            marker="s",
        )
        plt.text(row["Volatility (%)"] + 1, row["Return (compounded) (%)"], row["Portfolio"], fontsize=12)

    plt.title("United States\n1929-2016", fontsize=14, fontweight="bold", loc="center")
    plt.xlabel("Volatility (%)", fontsize=12)
    plt.ylabel("Average Compounded Return (%)", fontsize=12)
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.xlim(0, 40)
    plt.ylim(0, 20)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_output, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    end_date = pd.Timestamp(args.end_date)
    conservative_input = Path(args.conservative_input)
    speculative_input = Path(args.speculative_input)
    table_output = Path(args.table_output)
    plot_output = Path(args.plot_output)

    print(f"Loading conservative returns from {conservative_input}")
    conservative = load_stock_data(conservative_input)
    conservative = conservative[pd.to_datetime(conservative["Date"]) <= end_date]["Portfolio Returns"]
    print(f"Loading speculative returns from {speculative_input}")
    speculative = load_stock_data(speculative_input)
    speculative = speculative[pd.to_datetime(speculative["Date"]) <= end_date]["Portfolio Returns"]

    summary = pd.DataFrame(
        [
            {
                "Portfolio": "Conservative",
                "Return (compounded) (%)": annualized_return(conservative) * 100.0,
                "Volatility (%)": annualized_volatility(conservative) * 100.0,
            },
            {
                "Portfolio": "Speculative",
                "Return (compounded) (%)": annualized_return(speculative) * 100.0,
                "Volatility (%)": annualized_volatility(speculative) * 100.0,
            },
        ]
    )

    table_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_output, index=False)
    plot_summary(summary, plot_output)

    print(f"Saved Exhibit 2 table to {table_output}")
    print(f"Saved Exhibit 2 plot to {plot_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
