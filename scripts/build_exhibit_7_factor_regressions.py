from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conservative_formula.exhibit7 import build_cms_panel, pivot_regression_results, run_factor_regressions

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

CONS_PATH = PROCESSED_DIR / "conservative_portfolio_returns.parquet"
SPEC_PATH = PROCESSED_DIR / "speculative_portfolio_returns.parquet"

PANEL_CONFIGS = [
    {
        "name": "panel_a_ff3_carhart",
        "path": PROCESSED_DIR / "ff3_mom_monthly.parquet",
        "factor_sets": [
            ["mktrf"],
            ["mktrf", "smb"],
            ["mktrf", "smb", "hml"],
            ["mktrf", "smb", "hml", "mom"],
        ],
        "rename_map": {"mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML", "mom": "MOM"},
        "title": "Panel A: Fama-French [1993] / Carhart [1997]",
    },
    {
        "name": "panel_b_ff5_mom",
        "path": PROCESSED_DIR / "ff5_mom_monthly.parquet",
        "factor_sets": [
            ["mktrf"],
            ["mktrf", "smb", "hml"],
            ["mktrf", "smb", "hml", "rmw", "cma"],
            ["mktrf", "smb", "hml", "rmw", "cma", "mom"],
        ],
        "rename_map": {"mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML", "rmw": "RMW", "cma": "CMA", "mom": "MOM"},
        "title": "Panel B: Fama-French [2015]",
    },
    {
        "name": "panel_c_hxz",
        "path": PROCESSED_DIR / "q5_monthly_2024.parquet",
        "factor_sets": [
            ["mktrf", "me", "ia", "roe"],
        ],
        "rename_map": {"mktrf": "Mkt-RF", "me": "ME", "ia": "IA", "roe": "ROE", "eg": "EG"},
        "title": "Panel C: Hou, Xue, and Zhang",
    },
    {
        "name": "panel_d_aqr",
        "path": PROCESSED_DIR / "aqr_panel_monthly.parquet",
        "factor_sets": [
            ["mktrf", "smb", "hml", "qmj", "bab"],
            ["mktrf", "smb", "hml", "qmj", "bab", "mom"],
        ],
        "rename_map": {"mktrf": "Mkt-RF", "smb": "SMB", "hml": "HML", "qmj": "QMJ", "bab": "BAB", "mom": "MOM"},
        "title": "Panel D: AQR",
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conservative = pd.read_parquet(CONS_PATH)
    speculative = pd.read_parquet(SPEC_PATH)

    combined_tidy: list[pd.DataFrame] = []
    panel_exports: list[tuple[str, pd.DataFrame]] = []

    for config in PANEL_CONFIGS:
        factor_frame = pd.read_parquet(config["path"])
        cms_frame = build_cms_panel(conservative, speculative, factor_frame)
        results = run_factor_regressions(cms_frame, y_column="CMS", factor_combinations=config["factor_sets"])

        pretty_results = results.copy()
        pretty_results["Factor"] = pretty_results["Factor"].replace(config["rename_map"])
        pretty_results["Model"] = pretty_results["Model"].replace(
            {" + ".join(k): " + ".join(config["rename_map"].get(x, x) for x in k) for k in config["factor_sets"]}
        )

        model_order = [" + ".join(config["rename_map"].get(x, x) for x in combo) for combo in config["factor_sets"]]
        presentation = pivot_regression_results(pretty_results, model_order=model_order)

        tidy_path = OUTPUT_DIR / f"exhibit_7_{config['name']}_tidy.csv"
        presentation_path = OUTPUT_DIR / f"exhibit_7_{config['name']}_presentation.csv"

        pretty_results.to_csv(tidy_path, index=False)
        presentation.to_csv(presentation_path, index=False)

        panel_label = config["title"]
        panel_print = presentation.copy()
        panel_print.insert(0, "Panel", "")
        panel_print.loc[0, "Panel"] = panel_label

        print(f"Saved {panel_label} tidy results to {tidy_path}")
        print(f"Saved {panel_label} presentation table to {presentation_path}")
        print(panel_print.to_string(index=False))
        print()

        pretty_results.insert(0, "Panel", panel_label)
        combined_tidy.append(pretty_results)
        panel_exports.append((config["name"][:31], pretty_results))

    if combined_tidy:
        all_results = pd.concat(combined_tidy, ignore_index=True)
        all_results.to_csv(OUTPUT_DIR / "exhibit_7_all_panels_tidy.csv", index=False)

    if panel_exports:
        with pd.ExcelWriter(OUTPUT_DIR / "exhibit_7_regressions.xlsx") as writer:
            for sheet_name, frame in panel_exports:
                frame.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    main()
