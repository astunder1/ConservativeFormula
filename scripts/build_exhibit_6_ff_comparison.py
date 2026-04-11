from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.exhibit6 import (
    compute_sharpe_table,
    ensure_yearmonth,
    load_ff_sorted_portfolio_processed,
    merge_formula_and_rf,
    plot_grouped_sharpe_panel,
    plot_simple_sharpe_panel,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

FORMULA_PATH = PROCESSED_DIR / "conservative_portfolio_returns.parquet"
RF_PATH = PROCESSED_DIR / "ff_factors_monthly.parquet"

START_PERIOD = pd.Period("1929-01", freq="M")
END_PERIOD = pd.Period("2016-12", freq="M")
START_PERIOD_2X4X4 = pd.Period("1963-06", freq="M")

SIZE_BM_CONFIG = {
    "name": "size_bm",
    "path": PROCESSED_DIR / "6_Portfolios_2x3_monthly.parquet",
    "start_period": START_PERIOD,
    "rename_map": {
        "SMALL LoBM": "SG",
        "ME1 BM2": "SN",
        "SMALL HiBM": "SV",
        "BIG LoBM": "BG",
        "ME2 BM2": "BN",
        "BIG HiBM": "BV",
    },
    "portfolio_columns": ["Formula", "SG", "SN", "SV", "BG", "BN", "BV"],
    "title": "Sharpe Ratios 1929-2016 versus 6 Size/BtM",
    "highlight": "SV",
    "kind": "simple",
}

SIZE_MOM_CONFIG = {
    "name": "size_mom",
    "path": PROCESSED_DIR / "6_Portfolios_ME_Prior_12_2_monthly.parquet",
    "start_period": START_PERIOD,
    "rename_map": {
        "SMALL LoPRIOR": "SL",
        "ME1 PRIOR2": "SN",
        "SMALL HiPRIOR": "SW",
        "BIG LoPRIOR": "BL",
        "ME2 PRIOR2": "BN",
        "BIG HiPRIOR": "BW",
    },
    "portfolio_columns": ["Formula", "SL", "SN", "SW", "BL", "BN", "BW"],
    "title": "Sharpe Ratios 1929-2016 versus 6 Size/Mom",
    "highlight": "SW",
    "kind": "simple",
}

GROUP_MAIN = [
    ("Formula", 1),
    ("Small Growth", 4),
    ("Small BM2", 4),
    ("Small BM3", 4),
    ("Small Value", 4),
    ("Big Growth", 4),
    ("Big BM2", 4),
    ("Big BM3", 4),
    ("Big Value", 4),
]

INV_CONFIG = {
    "name": "beme_inv",
    "path": PROCESSED_DIR / "32_Portfolios_ME_BEME_INV_2x4x4_monthly.parquet",
    "start_period": START_PERIOD_2X4X4,
    "rename_map": None,
    "portfolio_columns": None,
    "title": "Sharpe Ratios 1963-2016 versus Size/BtM/Investments",
    "kind": "grouped",
    "sub_groups": ["Formula"] + ["Low Inv", "2", "3", "High Inv"] * 8,
    "main_groups": GROUP_MAIN,
}

OP_CONFIG = {
    "name": "beme_op",
    "path": PROCESSED_DIR / "32_Portfolios_ME_BEME_OP_2x4x4_monthly.parquet",
    "start_period": START_PERIOD_2X4X4,
    "rename_map": None,
    "portfolio_columns": None,
    "title": "Sharpe Ratios 1963-2016 versus Size/BtM/Profitability",
    "kind": "grouped",
    "sub_groups": ["Formula"] + ["Low Prof", "2", "3", "High Prof"] * 8,
    "main_groups": GROUP_MAIN,
}

CONFIGS = [SIZE_BM_CONFIG, SIZE_MOM_CONFIG, INV_CONFIG, OP_CONFIG]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    formula_returns = ensure_yearmonth(pd.read_parquet(FORMULA_PATH))
    rf_df = ensure_yearmonth(pd.read_parquet(RF_PATH), date_col="Date")

    for config in CONFIGS:
        ff_df = load_ff_sorted_portfolio_processed(
            path=config["path"],
            start_period=config["start_period"],
            end_period=END_PERIOD,
            rename_map=config["rename_map"],
        )
        raw_df, excess_df = merge_formula_and_rf(formula_returns, ff_df, rf_df)
        portfolio_columns = config["portfolio_columns"] or [col for col in raw_df.columns if col != "YearMonth"]
        sharpe_df = compute_sharpe_table(raw_df, excess_df, portfolio_columns=portfolio_columns)

        csv_out = OUTPUT_DIR / f"exhibit_6_{config['name']}_sharpes.csv"
        png_out = OUTPUT_DIR / f"exhibit_6_{config['name']}_sharpes.png"
        sharpe_df.to_csv(csv_out)

        if config["kind"] == "simple":
            plot_simple_sharpe_panel(
                sharpe_df,
                title=config["title"],
                output_path=png_out,
                highlight_column=config["highlight"],
            )
        else:
            plot_grouped_sharpe_panel(
                sharpe_df,
                title=config["title"],
                output_path=png_out,
                sub_groups=config["sub_groups"],
                main_groups=config["main_groups"],
            )

        print(f"Saved {config['name']} Sharpe table to {csv_out}")
        print(f"Saved {config['name']} chart to {png_out}")
        print(sharpe_df.to_string(float_format=lambda x: f"{x:.4f}"))
        print()


if __name__ == "__main__":
    main()
