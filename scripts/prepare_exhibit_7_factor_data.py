from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.exhibit7 import (
    build_aqr_panel,
    build_ff3_mom_panel,
    build_ff5_mom_panel,
    parse_ken_french_monthly_csv,
    prepare_bab_us_monthly,
    prepare_q5_monthly,
    prepare_qmj_us_monthly,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def save_frame(frame, stem: str) -> None:
    parquet_path = PROCESSED_DIR / f"{stem}.parquet"
    csv_path = PROCESSED_DIR / f"{stem}.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    print(f"Saved {stem} to {parquet_path}")
    print(f"Saved {stem} to {csv_path}")
    if "Date" in frame.columns:
        print(f"Rows: {len(frame):,} | Date range: {frame['Date'].min().date()} to {frame['Date'].max().date()}")
    else:
        print(f"Rows: {len(frame):,}")
    print()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ff3 = parse_ken_french_monthly_csv(RAW_DIR / "F-F_Research_Data_Factors.csv")
    ff5 = parse_ken_french_monthly_csv(RAW_DIR / "F-F_Research_Data_5_Factors_2x3.csv")
    mom = parse_ken_french_monthly_csv(RAW_DIR / "F-F_Momentum_Factor.csv")
    q5 = prepare_q5_monthly(RAW_DIR / "q5_factors_monthly_2024.csv")
    bab = prepare_bab_us_monthly(RAW_DIR / "Betting Against Beta Equity Factors Monthly.xlsx")
    qmj = prepare_qmj_us_monthly(RAW_DIR / "Quality Minus Junk Six Portfolios Formed on Size and Quality Monthly.xlsx")

    ff3_mom = build_ff3_mom_panel(ff3, mom)
    ff5_mom = build_ff5_mom_panel(ff5, mom)
    aqr = build_aqr_panel(ff3_mom, bab, qmj)

    save_frame(ff3, "ff3_monthly")
    save_frame(ff5, "ff5_monthly")
    save_frame(mom, "ff_momentum_monthly")
    save_frame(q5, "q5_monthly_2024")
    save_frame(bab, "bab_us_monthly")
    save_frame(qmj, "qmj_us_monthly")
    save_frame(ff3_mom, "ff3_mom_monthly")
    save_frame(ff5_mom, "ff5_mom_monthly")
    save_frame(aqr, "aqr_panel_monthly")


if __name__ == "__main__":
    main()
