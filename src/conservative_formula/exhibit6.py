from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW_ROWS_DEFAULT = 1170
RAW_ROWS_2X4X4 = 735


def ensure_yearmonth(frame: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = frame.copy()
    if "YearMonth" not in out.columns:
        out["YearMonth"] = pd.to_datetime(out[date_col]).dt.to_period("M")
    elif not isinstance(out["YearMonth"].dtype, pd.PeriodDtype):
        out["YearMonth"] = pd.PeriodIndex(out["YearMonth"], freq="M")
    return out


def annualized_simple_return(returns: pd.Series) -> float:
    return float(returns.mean() * 12)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * (12 ** 0.5))


def sharpe_ratio_simple(excess_returns: pd.Series, raw_returns: pd.Series) -> float:
    vol = annualized_volatility(raw_returns)
    return float(annualized_simple_return(excess_returns) / vol) if vol else float("nan")


def load_ff_sorted_portfolio_csv(
    path: Path,
    start_period: pd.Period,
    end_period: pd.Period,
    rename_map: dict[str, str] | None = None,
    end_row: int = RAW_ROWS_DEFAULT,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.iloc[:end_row].copy()
    frame = frame.rename(columns={"Unnamed: 0": "Date"}).dropna(how="all")
    frame["Date"] = pd.to_datetime(frame["Date"].astype(str), format="%Y%m", errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["YearMonth"] = frame["Date"].dt.to_period("M")
    frame = frame.drop(columns=["Date"])

    value_columns = [col for col in frame.columns if col != "YearMonth"]
    frame.loc[:, value_columns] = frame.loc[:, value_columns].apply(pd.to_numeric, errors="coerce") / 100.0

    if rename_map:
        frame = frame.rename(columns=rename_map)

    frame = frame.loc[(frame["YearMonth"] > start_period) & (frame["YearMonth"] <= end_period)].copy()
    return frame


def load_ff_sorted_portfolio_processed(
    path: Path,
    start_period: pd.Period,
    end_period: pd.Period,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    frame = ensure_yearmonth(frame, date_col="Date")

    value_columns = [col for col in frame.columns if col not in {"Date", "YearMonth"}]
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if rename_map:
        frame = frame.rename(columns=rename_map)

    frame = frame.loc[(frame["YearMonth"] > start_period) & (frame["YearMonth"] <= end_period)].copy()
    keep_columns = ["YearMonth"] + [col for col in frame.columns if col not in {"Date", "YearMonth"}]
    return frame[keep_columns]


def merge_formula_and_rf(
    formula_returns: pd.DataFrame,
    ff_portfolios: pd.DataFrame,
    rf_df: pd.DataFrame,
    rf_column: str = "rf",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    formula = ensure_yearmonth(formula_returns).copy()
    formula = formula.rename(columns={"Portfolio Returns": "Formula"})

    merged = pd.merge(formula[["YearMonth", "Formula"]], ff_portfolios, on="YearMonth", how="inner")
    merged_rf = pd.merge(merged, rf_df[["YearMonth", rf_column]], on="YearMonth", how="inner")

    excess = merged_rf.copy()
    return_columns = [col for col in excess.columns if col not in ["YearMonth", rf_column]]
    excess.loc[:, return_columns] = excess.loc[:, return_columns].subtract(excess[rf_column], axis=0)
    excess = excess.drop(columns=[rf_column])

    raw = merged_rf.drop(columns=[rf_column])
    return raw, excess


def compute_sharpe_table(
    raw_df: pd.DataFrame,
    excess_df: pd.DataFrame,
    portfolio_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = portfolio_columns or [col for col in raw_df.columns if col != "YearMonth"]
    sharpe_dict: dict[str, float] = {}
    for column in columns:
        sharpe_dict[column] = sharpe_ratio_simple(excess_df[column], raw_df[column])
    return pd.DataFrame.from_dict(sharpe_dict, orient="index", columns=["Sharpe Ratio"])


def plot_simple_sharpe_panel(
    sharpe_df: pd.DataFrame,
    title: str,
    output_path: Path,
    highlight_column: str | None = None,
    formula_color: str = "#00bfff",
    default_color: str = "gray",
    highlight_color: str = "black",
    ylim: tuple[float, float] = (0.0, 0.9),
) -> None:
    colors = [formula_color if idx == "Formula" else default_color for idx in sharpe_df.index]
    if highlight_column and highlight_column in sharpe_df.index:
        colors[sharpe_df.index.get_loc(highlight_column)] = highlight_color

    plt.figure(figsize=(7, 5))
    plt.bar(sharpe_df.index, sharpe_df["Sharpe Ratio"], color=colors, width=0.4)
    plt.title(title, fontsize=16, weight="bold")
    plt.ylabel("Sharpe Ratio", fontsize=14)
    plt.xlabel("Portfolio", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(*ylim)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_grouped_sharpe_panel(
    sharpe_df: pd.DataFrame,
    title: str,
    output_path: Path,
    sub_groups: list[str],
    main_groups: list[tuple[str, int]],
    highlight_mode: str = "max",
    ylim: tuple[float, float] = (0.0, 0.9),
) -> None:
    if highlight_mode == "max":
        highlight_index = sharpe_df["Sharpe Ratio"].idxmax()
        highlight_position = sharpe_df.index.get_loc(highlight_index)
    else:
        highlight_position = None

    bar_colors = ["#00bfff"] + ["gray"] * (len(sharpe_df) - 1)
    if highlight_position is not None:
        bar_colors[highlight_position] = "black"

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(sharpe_df))
    ax.bar(x, sharpe_df["Sharpe Ratio"], color=bar_colors, width=0.36)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub_groups, fontsize=10, rotation=90)

    current_position = 0
    for group_name, span in main_groups:
        ax.text(current_position + span / 2 - 0.5, -0.18, group_name, ha="center", va="top", fontsize=10, fontweight="bold")
        current_position += span

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
