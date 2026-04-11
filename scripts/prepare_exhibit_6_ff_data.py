from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

FILES = [
    '6_Portfolios_2x3.csv',
    '6_Portfolios_ME_Prior_12_2.csv',
    '32_Portfolios_ME_BEME_INV_2x4x4.csv',
    '32_Portfolios_ME_BEME_OP_2x4x4.csv',
]


def find_monthly_header(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(',') and stripped.count(',') >= 2:
            return idx
        first_field = line.split(',', 1)[0].strip().strip('"').lower()
        if first_field in {'date', 'unnamed: 0'}:
            return idx
    raise ValueError('Could not find a monthly data header row.')


def parse_ff_portfolio_file(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    header_idx = find_monthly_header(lines)
    raw_header = [part.strip() for part in lines[header_idx].split(',')]
    if raw_header and raw_header[0] == '':
        raw_header[0] = 'DateRaw'
    header = raw_header

    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break

        first_field = stripped.split(',', 1)[0].strip().strip('"')
        if not (first_field.isdigit() and len(first_field) == 6):
            break

        parts = [part.strip() for part in stripped.split(',')]
        parts = parts[: len(header)] + [''] * max(0, len(header) - len(parts))
        rows.append(parts[: len(header)])

    if not rows:
        raise ValueError(f'No monthly rows parsed from {path.name}')

    df = pd.DataFrame(rows, columns=header)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: 'DateRaw'})
    df['Date'] = pd.to_datetime(df['DateRaw'], format='%Y%m', errors='coerce')
    df = df.dropna(subset=['Date']).copy()
    df['YearMonth'] = df['Date'].dt.to_period('M')

    value_columns = [col for col in df.columns if col not in {'DateRaw', 'Date', 'YearMonth'}]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce') / 100.0

    df = df.drop(columns=['DateRaw'])
    ordered_columns = ['Date', 'YearMonth'] + value_columns
    return df[ordered_columns]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        raw_path = RAW_DIR / name
        cleaned = parse_ff_portfolio_file(raw_path)
        stem = raw_path.stem
        parquet_path = PROCESSED_DIR / f'{stem}_monthly.parquet'
        csv_path = PROCESSED_DIR / f'{stem}_monthly.csv'
        cleaned.to_parquet(parquet_path, index=False)
        cleaned.to_csv(csv_path, index=False)
        print(f'Saved cleaned monthly data to {parquet_path}')
        print(f'Saved cleaned monthly data to {csv_path}')
        print(f"Rows: {len(cleaned):,} | Date range: {cleaned['Date'].min().date()} to {cleaned['Date'].max().date()}")
        print()


if __name__ == '__main__':
    main()
