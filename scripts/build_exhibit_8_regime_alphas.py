from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.exhibit8 import build_regime_analysis_frame, plot_regime_alphas, summarize_regime_alphas

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

CONS_PATH = PROCESSED_DIR / "conservative_portfolio_returns.parquet"
FF3_PATH = PROCESSED_DIR / "ff3_monthly.parquet"
REGIME_PATH = PROCESSED_DIR / "economic_regimes_monthly.parquet"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conservative = pd.read_parquet(CONS_PATH)
    ff3 = pd.read_parquet(FF3_PATH)
    regimes = pd.read_parquet(REGIME_PATH)

    analysis_frame = build_regime_analysis_frame(conservative, ff3, regimes)
    summary = summarize_regime_alphas(analysis_frame)

    analysis_path = OUTPUT_DIR / "exhibit_8_regime_analysis_panel.csv"
    summary_path = OUTPUT_DIR / "exhibit_8_regime_alphas.csv"
    chart_path = OUTPUT_DIR / "exhibit_8_regime_alphas.png"

    analysis_frame.to_csv(analysis_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_regime_alphas(summary, chart_path)

    print(f"Saved Exhibit 8 analysis panel to {analysis_path}")
    print(f"Saved Exhibit 8 summary table to {summary_path}")
    print(f"Saved Exhibit 8 chart to {chart_path}")
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == "__main__":
    main()
