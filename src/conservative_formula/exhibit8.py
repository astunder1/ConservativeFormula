from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter


def ensure_yearmonth(frame: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = frame.copy()
    if "YearMonth" not in out.columns:
        out["YearMonth"] = pd.to_datetime(out[date_col]).dt.to_period("M")
    elif not isinstance(out["YearMonth"].dtype, pd.PeriodDtype):
        out["YearMonth"] = pd.PeriodIndex(out["YearMonth"], freq="M")
    return out


def load_fred_monthly_csv(path: Path, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    frame["Date"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["YearMonth"] = frame["Date"].dt.to_period("M")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    return frame[["Date", "YearMonth", value_column]]


def classify_high_low(series: pd.Series) -> pd.Series:
    median_value = series.median()
    return pd.Series(np.where(series > median_value, "High", "Low"), index=series.index)


def classify_rising_falling(series: pd.Series, window: int = 3) -> pd.Series:
    rolling_avg = series.rolling(window=window, min_periods=window).mean()
    return pd.Series(np.where(series > rolling_avg, "Rising", "Falling"), index=series.index)


def prepare_regime_frame(usrec: pd.DataFrame, gs10: pd.DataFrame, aaa: pd.DataFrame, baa: pd.DataFrame) -> pd.DataFrame:
    regime = pd.merge(usrec[["YearMonth", "USREC"]], gs10[["YearMonth", "GS10"]], on="YearMonth", how="outer")
    regime = pd.merge(regime, aaa[["YearMonth", "AAA"]], on="YearMonth", how="outer")
    regime = pd.merge(regime, baa[["YearMonth", "BAA"]], on="YearMonth", how="outer")
    regime = regime.sort_values("YearMonth").reset_index(drop=True)
    regime["Date"] = regime["YearMonth"].dt.to_timestamp("M")

    regime["CreditSpread"] = regime["BAA"] - regime["AAA"]
    regime["GS10_HighLow"] = classify_high_low(regime["GS10"])
    regime["GS10_Direction"] = classify_rising_falling(regime["GS10"])
    regime["Spread_HighLow"] = classify_high_low(regime["CreditSpread"])
    regime["Spread_Direction"] = classify_rising_falling(regime["CreditSpread"])

    return regime[
        [
            "Date",
            "YearMonth",
            "USREC",
            "GS10",
            "AAA",
            "BAA",
            "CreditSpread",
            "GS10_HighLow",
            "GS10_Direction",
            "Spread_HighLow",
            "Spread_Direction",
        ]
    ]


def capm_alpha_beta(frame: pd.DataFrame, excess_return_col: str, market_excess_col: str) -> tuple[float, float, int]:
    clean = frame[[excess_return_col, market_excess_col]].dropna()
    x = sm.add_constant(clean[market_excess_col])
    model = sm.OLS(clean[excess_return_col], x).fit()
    return float(model.params["const"]), float(model.params[market_excess_col]), int(model.nobs)


def build_regime_analysis_frame(conservative_returns: pd.DataFrame, ff3: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    cons = ensure_yearmonth(conservative_returns).rename(columns={"Portfolio Returns": "Conservative"})
    ff = ensure_yearmonth(ff3, date_col="Date").rename(columns={"Mkt-RF": "mktrf", "RF": "rf"})
    merged = pd.merge(cons[["YearMonth", "Conservative"]], ff[["YearMonth", "mktrf", "rf"]], on="YearMonth", how="inner")
    merged = pd.merge(merged, regime, on="YearMonth", how="inner")
    merged["ExcessConservative"] = merged["Conservative"] - merged["rf"]
    return merged


def summarize_regime_alphas(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    regime_specs = [
        ("All", frame),
        ("Expansion", frame.loc[frame["USREC"] == 0]),
        ("Recession", frame.loc[frame["USREC"] == 1]),
        ("Low 10Y", frame.loc[frame["GS10_HighLow"] == "Low"]),
        ("High 10Y", frame.loc[frame["GS10_HighLow"] == "High"]),
        ("Falling 10Y", frame.loc[frame["GS10_Direction"] == "Falling"]),
        ("Rising 10Y", frame.loc[frame["GS10_Direction"] == "Rising"]),
        ("Low Spread", frame.loc[frame["Spread_HighLow"] == "Low"]),
        ("High Spread", frame.loc[frame["Spread_HighLow"] == "High"]),
        ("Falling Spread", frame.loc[frame["Spread_Direction"] == "Falling"]),
        ("Rising Spread", frame.loc[frame["Spread_Direction"] == "Rising"]),
    ]

    for label, subset in regime_specs:
        alpha, beta, nobs = capm_alpha_beta(subset, excess_return_col="ExcessConservative", market_excess_col="mktrf")
        rows.append(
            {
                "Regime": label,
                "AlphaMonthly": alpha,
                "AlphaAnnualized": alpha * 12.0,
                "Beta": beta,
                "Nobs": nobs,
            }
        )

    return pd.DataFrame(rows)


def plot_regime_alphas(summary: pd.DataFrame, output_path: Path) -> None:
    order = [
        "All",
        "Expansion",
        "Recession",
        "Low 10Y",
        "High 10Y",
        "Falling 10Y",
        "Rising 10Y",
        "Low Spread",
        "High Spread",
        "Falling Spread",
        "Rising Spread",
    ]
    plot_df = summary.set_index("Regime").loc[order].reset_index()
    alphas_pct = plot_df["AlphaAnnualized"] * 100.0

    sub_groups = ["All", "Expansion", "Recession", "Low", "High", "Falling", "Rising", "Low", "High", "Falling", "Rising"]
    main_groups = [("All", 1), ("NBER", 2), ("10Y Bond Yield", 4), ("Credit Spread", 4)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(alphas_pct))
    ax.bar(x, alphas_pct, color="gray", width=0.5, edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub_groups, fontsize=10, rotation=90)

    current_position = 0
    for group_name, span in main_groups:
        ax.text(current_position + span / 2 - 0.5, -0.22, group_name, ha="center", va="top", fontsize=10, fontweight="bold")
        current_position += span

    ax.set_title("Conservative Formula Alpha Across Economic Regimes", fontsize=16, weight="bold")
    ax.set_ylabel("Alpha", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
