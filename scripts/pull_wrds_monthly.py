from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import wrds
except ImportError:
    print("Missing package 'wrds'. Install it with: pip install wrds")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = RAW_DIR / "crsp_us_monthly.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull a paper-oriented monthly CRSP panel for the Conservative Formula replication."
    )
    parser.add_argument("--wrds-username", default=None)
    parser.add_argument("--start-date", default="1925-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SQL and exit without connecting.",
    )
    return parser.parse_args()


def get_wrds_username(cli_username: str | None) -> str:
    username = cli_username or os.getenv("WRDS_USERNAME")
    if not username:
        raise ValueError(
            "WRDS username not provided. Pass --wrds-username or set WRDS_USERNAME."
        )
    return username


def build_main_sql(start_date: str, end_date: str) -> str:
    return f"""
        WITH delist_monthly AS (
            SELECT
                permno,
                DATE_TRUNC('month', dlstdt)::date + INTERVAL '1 month' - INTERVAL '1 day' AS date,
                dlret,
                dlretx,
                dlstcd,
                nwperm
            FROM crsp.msedelist
            WHERE dlstdt BETWEEN '{start_date}'::date AND '{end_date}'::date
        ),
        dist_monthly AS (
            SELECT
                permno,
                DATE_TRUNC('month', exdt)::date + INTERVAL '1 month' - INTERVAL '1 day' AS date,
                SUM(COALESCE(divamt, 0.0)) AS divamt
            FROM crsp.msedist
            WHERE exdt BETWEEN '{start_date}'::date AND '{end_date}'::date
            GROUP BY permno, DATE_TRUNC('month', exdt)
        )
        SELECT
            a.permno AS "PERMNO",
            a.permco AS "PERMCO",
            a.date AS "Date",
            a.ret AS "RET",
            a.retx AS "RETX",
            a.prc AS "PRC",
            a.altprc AS "ALTPRC",
            a.shrout AS "SHROUT",
            a.cfacshr AS "CFACSHR",
            b.exchcd AS "EXCHCD",
            b.shrcd AS "SHRCD",
            b.ticker AS "TICKER",
            b.comnam AS "COMNAM",
            c.dlret AS "DLRET",
            c.dlretx AS "DLRETX",
            c.dlstcd AS "DLSTCD",
            c.nwperm AS "NWPERM",
            COALESCE(d.divamt, 0.0) AS "DIVAMT"
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
           AND b.namedt <= a.date
           AND a.date <= b.nameendt
        LEFT JOIN delist_monthly AS c
            ON a.permno = c.permno
           AND a.date = c.date
        LEFT JOIN dist_monthly AS d
            ON a.permno = d.permno
           AND a.date = d.date
        WHERE a.date BETWEEN '{start_date}'::date AND '{end_date}'::date
        ORDER BY a.date, a.permno
    """


def clean_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")

    numeric_columns = [
        "PERMNO",
        "PERMCO",
        "RET",
        "RETX",
        "PRC",
        "ALTPRC",
        "SHROUT",
        "CFACSHR",
        "EXCHCD",
        "SHRCD",
        "DLRET",
        "DLRETX",
        "DLSTCD",
        "NWPERM",
        "DIVAMT",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    string_columns = ["TICKER", "COMNAM"]
    for column in string_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()

    return frame


def summarize(frame: pd.DataFrame) -> None:
    print(f"Rows: {len(frame):,}")
    print(f"Unique PERMNOs: {frame['PERMNO'].nunique():,}")
    print(f"Date range: {frame['Date'].min()} to {frame['Date'].max()}")
    print("Missing-rate snapshot:")
    print(
        frame[
            ["RET", "RETX", "ALTPRC", "SHROUT", "EXCHCD", "SHRCD", "DLRET", "DLSTCD", "DIVAMT"]
        ]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )


def main() -> None:
    args = parse_args()
    username = get_wrds_username(args.wrds_username)
    output_path = Path(args.output)
    sql = build_main_sql(args.start_date, args.end_date)

    if args.dry_run:
        print(f"WRDS username: {username}")
        print(f"Output path: {output_path}")
        print(sql)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to WRDS as {username} ...")
    db = wrds.Connection(wrds_username=username)
    try:
        frame = db.raw_sql(sql, date_cols=["Date"])
    finally:
        db.close()

    frame = clean_types(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    print(f"Saved monthly panel to {output_path}")
    summarize(frame)


if __name__ == "__main__":
    main()
