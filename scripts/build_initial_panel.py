from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.cleaning import prepare_initial_stock_panel
from conservative_formula.data_loading import load_stock_data


DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "crsp_us_monthly.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "crsp_us_monthly_initial.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the initial cleaned monthly panel for the Conservative Formula replication."
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
            ["RET ADJ", "ALTPRC", "SHROUT", "EXCHCD", "MKT Cap", "Lagged MKT Cap"]
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

    print(f"Loading raw panel from {input_path}")
    frame = load_stock_data(input_path)

    print("Applying initial cleaning pipeline...")
    cleaned = prepare_initial_stock_panel(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)

    print(f"Saved initial cleaned panel to {output_path}")
    summarize(cleaned)


if __name__ == "__main__":
    main()
