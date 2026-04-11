from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.exhibit8 import load_fred_monthly_csv, prepare_regime_frame

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def save_frame(frame, stem: str) -> None:
    parquet_path = PROCESSED_DIR / f"{stem}.parquet"
    csv_path = PROCESSED_DIR / f"{stem}.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    print(f"Saved {stem} to {parquet_path}")
    print(f"Saved {stem} to {csv_path}")
    print(f"Rows: {len(frame):,} | Date range: {frame['Date'].min().date()} to {frame['Date'].max().date()}")
    print()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    usrec = load_fred_monthly_csv(RAW_DIR / "USREC.csv", "USREC")
    gs10 = load_fred_monthly_csv(RAW_DIR / "GS10.csv", "GS10")
    aaa = load_fred_monthly_csv(RAW_DIR / "AAA.csv", "AAA")
    baa = load_fred_monthly_csv(RAW_DIR / "BAA.csv", "BAA")

    regime = prepare_regime_frame(usrec, gs10, aaa, baa)

    save_frame(usrec, "usrec_monthly")
    save_frame(gs10, "gs10_monthly")
    save_frame(aaa, "aaa_monthly")
    save_frame(baa, "baa_monthly")
    save_frame(regime, "economic_regimes_monthly")


if __name__ == "__main__":
    main()
