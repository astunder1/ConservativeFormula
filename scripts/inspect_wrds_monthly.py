from __future__ import annotations

import os
import sys

try:
    import wrds
except ImportError:
    print("Missing package 'wrds'. Install it with: pip install wrds")
    sys.exit(1)


def run_query(db: wrds.Connection, title: str, sql: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    try:
        df = db.raw_sql(sql)
        if df.empty:
            print("(no rows)")
        else:
            print(df.to_string(index=False))
    except Exception as exc:
        print(f"Query failed: {exc}")


def main() -> None:
    username = os.getenv("WRDS_USERNAME")
    if not username:
        print("Set WRDS_USERNAME first.")
        print('PowerShell example: $env:WRDS_USERNAME="your_wrds_username"')
        sys.exit(1)

    print(f"Connecting to WRDS as {username} ...")
    db = wrds.Connection(wrds_username=username)

    try:
        run_query(
            db,
            "Likely CRSP monthly / delist / dividend-related tables",
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = 'crsp'
              AND (
                table_name ILIKE '%msf%'
                OR table_name ILIKE '%mse%'
                OR table_name ILIKE '%delist%'
                OR table_name ILIKE '%dist%'
                OR table_name ILIKE '%div%'
              )
            ORDER BY table_name;
            """,
        )

        for table_name in ["msf", "msenames", "msedelist"]:
            run_query(
                db,
                f"Columns in crsp.{table_name}",
                f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'crsp'
                  AND table_name = '{table_name}'
                ORDER BY ordinal_position;
                """,
            )

        run_query(
            db,
            "Any CRSP columns containing div/dist/pay",
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'crsp'
              AND (
                column_name ILIKE '%div%'
                OR column_name ILIKE '%dist%'
                OR column_name ILIKE '%pay%'
              )
            ORDER BY table_name, column_name;
            """,
        )

        run_query(
            db,
            "Sample rows from crsp.msf",
            """
            SELECT *
            FROM crsp.msf
            LIMIT 5;
            """,
        )

        run_query(
            db,
            "Sample monthly join: msf + msenames",
            """
            SELECT
                a.permno,
                a.date,
                a.ret,
                a.retx,
                a.prc,
                a.shrout,
                b.exchcd,
                b.shrcd
            FROM crsp.msf AS a
            LEFT JOIN crsp.msenames AS b
                ON a.permno = b.permno
               AND b.namedt <= a.date
               AND a.date <= b.nameendt
            LIMIT 20;
            """,
        )

        run_query(
            db,
            "Sample rows from crsp.msedelist",
            """
            SELECT *
            FROM crsp.msedelist
            LIMIT 10;
            """,
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
