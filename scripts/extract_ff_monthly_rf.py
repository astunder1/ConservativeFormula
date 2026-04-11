from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "F-F_Research_Data_Factors.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ff_factors_monthly.parquet"


def _find_monthly_header(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        normalized = line.strip().lower().replace(" ", "")
        if normalized.startswith(",mkt-rf,smb,hml,rf"):
            return idx
    raise ValueError("Could not find the monthly Fama-French header row.")


def load_ff_monthly_factors(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = _find_monthly_header(lines)

    data_rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break

        first_field = stripped.split(",", 1)[0].strip()
        if not (first_field.isdigit() and len(first_field) == 6):
            break

        data_rows.append([part.strip() for part in stripped.split(",")[:5]])

    if not data_rows:
        raise ValueError("No monthly factor rows were parsed from the source file.")

    df = pd.DataFrame(data_rows, columns=["yyyymm", "mktrf", "smb", "hml", "rf"])
    df["Date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df["YearMonth"] = df["Date"].dt.to_period("M")

    for column in ["mktrf", "smb", "hml", "rf"]:
        df[column] = pd.to_numeric(df[column], errors="coerce") / 100.0

    return df[["Date", "YearMonth", "mktrf", "smb", "hml", "rf"]].dropna().reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract monthly Fama-French factors and RF from raw Kenneth French CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to F-F_Research_Data_Factors.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to cleaned parquet output")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional path to also save cleaned CSV output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_path = args.output.resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    df = load_ff_monthly_factors(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    if args.csv_output is not None:
        csv_output = args.csv_output.resolve()
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_output, index=False)

    print(f"Saved monthly FF factors to {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
