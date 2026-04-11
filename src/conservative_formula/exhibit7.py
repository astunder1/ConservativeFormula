from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_yearmonth(frame: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = frame.copy()
    if "YearMonth" not in out.columns:
        out["YearMonth"] = pd.to_datetime(out[date_col]).dt.to_period("M")
    elif not isinstance(out["YearMonth"].dtype, pd.PeriodDtype):
        out["YearMonth"] = pd.PeriodIndex(out["YearMonth"], freq="M")
    return out


def find_ken_french_header(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(",") and stripped.count(",") >= 1:
            return idx
    raise ValueError("Could not find Kenneth French monthly header row.")


def parse_ken_french_monthly_csv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = find_ken_french_header(lines)

    header = [part.strip() for part in lines[header_idx].split(",")]
    if header and header[0] == "":
        header[0] = "DateRaw"

    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break

        first_field = stripped.split(",", 1)[0].strip().strip('"')
        if not (first_field.isdigit() and len(first_field) == 6):
            break

        parts = [part.strip() for part in stripped.split(",")]
        parts = parts[: len(header)] + [""] * max(0, len(header) - len(parts))
        rows.append(parts[: len(header)])

    if not rows:
        raise ValueError(f"No monthly rows parsed from {path.name}")

    frame = pd.DataFrame(rows, columns=header).rename(columns={header[0]: "DateRaw"})
    frame["Date"] = pd.to_datetime(frame["DateRaw"], format="%Y%m", errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["YearMonth"] = frame["Date"].dt.to_period("M")

    value_columns = [col for col in frame.columns if col not in {"DateRaw", "Date", "YearMonth"}]
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0

    return frame[["Date", "YearMonth"] + value_columns]


def prepare_q5_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame[["year", "month"]].assign(day=1))
    frame["YearMonth"] = frame["Date"].dt.to_period("M")

    rename_map = {
        "R_F": "rf",
        "R_MKT": "mktrf",
        "R_ME": "me",
        "R_IA": "ia",
        "R_ROE": "roe",
        "R_EG": "eg",
    }
    frame = frame.rename(columns=rename_map)

    value_columns = ["rf", "mktrf", "me", "ia", "roe", "eg"]
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0

    return frame[["Date", "YearMonth"] + value_columns]


def prepare_bab_us_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="BAB Factors", header=18)
    frame = frame.rename(columns={"DATE": "Date", "USA": "bab"})
    frame = frame[["Date", "bab"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["YearMonth"] = frame["Date"].dt.to_period("M")
    frame["bab"] = pd.to_numeric(frame["bab"], errors="coerce")
    return frame[["Date", "YearMonth", "bab"]]


def prepare_qmj_us_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Size x Quality (2 x3)", header=18)
    frame = frame.rename(columns={"DATE": "Date", "Factor": "qmj"})
    frame = frame[["Date", "qmj"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["YearMonth"] = frame["Date"].dt.to_period("M")
    frame["qmj"] = pd.to_numeric(frame["qmj"], errors="coerce")
    return frame[["Date", "YearMonth", "qmj"]]


def build_ff3_mom_panel(ff3: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(ff3, mom[["YearMonth", "Mom"]], on="YearMonth", how="inner")
    merged = merged.rename(columns={"Mom": "mom"})
    return merged[["Date", "YearMonth", "Mkt-RF", "SMB", "HML", "RF", "mom"]].rename(
        columns={"Mkt-RF": "mktrf", "SMB": "smb", "HML": "hml", "RF": "rf"}
    )


def build_ff5_mom_panel(ff5: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(ff5, mom[["YearMonth", "Mom"]], on="YearMonth", how="inner")
    merged = merged.rename(columns={"Mom": "mom"})
    return merged[["Date", "YearMonth", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "mom"]].rename(
        columns={"Mkt-RF": "mktrf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma", "RF": "rf"}
    )


def build_aqr_panel(ff3_mom: pd.DataFrame, bab: pd.DataFrame, qmj: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(ff3_mom, bab[["YearMonth", "bab"]], on="YearMonth", how="inner")
    merged = pd.merge(merged, qmj[["YearMonth", "qmj"]], on="YearMonth", how="inner")
    return merged


def build_cms_panel(conservative_returns: pd.DataFrame, speculative_returns: pd.DataFrame, factor_frame: pd.DataFrame) -> pd.DataFrame:
    cons = ensure_yearmonth(conservative_returns).rename(columns={"Portfolio Returns": "Conservative"})
    spec = ensure_yearmonth(speculative_returns).rename(columns={"Portfolio Returns": "Speculative"})

    cms = pd.merge(cons[["YearMonth", "Conservative"]], spec[["YearMonth", "Speculative"]], on="YearMonth", how="inner")
    cms["CMS"] = cms["Conservative"] - cms["Speculative"]
    return pd.merge(cms, ensure_yearmonth(factor_frame), on="YearMonth", how="inner")


def run_factor_regressions(frame: pd.DataFrame, y_column: str, factor_combinations: list[list[str]]) -> pd.DataFrame:
    import statsmodels.api as sm

    results: list[dict[str, float | str]] = []
    y = frame[y_column]

    for factors in factor_combinations:
        x = sm.add_constant(frame[factors])
        model = sm.OLS(y, x, missing="drop").fit()
        model_name = " + ".join(factors)

        for param in model.params.index:
            factor_name = "Alpha" if param == "const" else param
            coefficient = float(model.params[param] * 12) if param == "const" else float(model.params[param])
            results.append(
                {
                    "Model": model_name,
                    "Factor": factor_name,
                    "Coefficient": coefficient,
                    "t-value": float(model.tvalues[param]),
                    "Nobs": float(model.nobs),
                    "R2": float(model.rsquared),
                }
            )

    return pd.DataFrame(results)


def pivot_regression_results(results_df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    factor_order = ["Alpha", "Mkt-RF", "SMB", "HML", "MOM", "RMW", "CMA", "ME", "IA", "ROE", "EG", "QMJ", "BAB"]

    for model_name in model_order:
        subset = results_df.loc[results_df["Model"] == model_name].copy()
        coef_row: dict[str, str] = {"Metric": f"{model_name} Coef"}
        t_row: dict[str, str] = {"Metric": f"{model_name} t"}

        for factor in factor_order:
            hit = subset.loc[subset["Factor"] == factor]
            if not hit.empty:
                coef_row[factor] = f"{hit['Coefficient'].iloc[0]:.3f}"
                t_row[factor] = f"{hit['t-value'].iloc[0]:.3f}"
            else:
                coef_row[factor] = ""
                t_row[factor] = ""

        rows.append(coef_row)
        rows.append(t_row)

    presentation = pd.DataFrame(rows)
    nonempty_columns = ["Metric"] + [col for col in presentation.columns if col != "Metric" and presentation[col].ne("").any()]
    return presentation[nonempty_columns]
