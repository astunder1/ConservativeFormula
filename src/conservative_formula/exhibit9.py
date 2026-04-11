from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from conservative_formula.evaluation import annualized_return


def ensure_yearmonth(frame: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = frame.copy()
    if "YearMonth" not in out.columns:
        out["YearMonth"] = pd.to_datetime(out[date_col]).dt.to_period("M")
    elif not isinstance(out["YearMonth"].dtype, pd.PeriodDtype):
        out["YearMonth"] = pd.PeriodIndex(out["YearMonth"], freq="M")
    return out


def compute_quarterly_turnover(selection_frame: pd.DataFrame, portfolio_label: str = "Conservative") -> tuple[pd.DataFrame, float]:
    frame = selection_frame.loc[selection_frame["Portfolio"] == portfolio_label].copy()
    frame["RebalanceDate"] = pd.to_datetime(frame["RebalanceDate"])

    grouped = frame.groupby("RebalanceDate")["PERMNO"].apply(lambda x: set(x.dropna().astype(int)))
    if grouped.empty or len(grouped) < 2:
        return pd.DataFrame(columns=["Current Quarter", "Next Quarter", "Overlap Count", "Turnover Count", "Turnover Fraction"]), float("nan")

    portfolio_size = float(grouped.apply(len).median())
    rows: list[dict[str, float | pd.Timestamp]] = []
    for current_date, next_date in zip(grouped.index[:-1], grouped.index[1:]):
        current_stocks = grouped.loc[current_date]
        next_stocks = grouped.loc[next_date]
        overlap_count = len(current_stocks.intersection(next_stocks))
        turnover_count = portfolio_size - overlap_count
        turnover_fraction = turnover_count / portfolio_size if portfolio_size else float("nan")
        rows.append(
            {
                "Current Quarter": current_date,
                "Next Quarter": next_date,
                "Overlap Count": float(overlap_count),
                "Turnover Count": float(turnover_count),
                "Turnover Fraction": float(turnover_fraction),
            }
        )

    turnover_df = pd.DataFrame(rows)
    avg_turnover_fraction = float(turnover_df["Turnover Fraction"].mean()) if not turnover_df.empty else float("nan")
    return turnover_df, avg_turnover_fraction


def build_trading_cost_summary(
    conservative_returns: pd.DataFrame,
    selections: pd.DataFrame,
    start_period: pd.Period | None = None,
    end_period: pd.Period | None = None,
    low_cost_bps: float = 10.0,
    high_cost_bps: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = ensure_yearmonth(conservative_returns)
    sel = selections.copy()
    sel["YearMonth"] = pd.to_datetime(sel["RebalanceDate"]).dt.to_period("M")

    if start_period is not None:
        returns = returns.loc[returns["YearMonth"] >= start_period].copy()
        sel = sel.loc[sel["YearMonth"] >= start_period].copy()
    if end_period is not None:
        returns = returns.loc[returns["YearMonth"] <= end_period].copy()
        sel = sel.loc[sel["YearMonth"] <= end_period].copy()

    turnover_df, avg_turnover_fraction = compute_quarterly_turnover(sel, portfolio_label="Conservative")
    gross_return = annualized_return(returns["Portfolio Returns"], periods_per_year=12)

    low_cost_decimal = (low_cost_bps / 10000.0) * avg_turnover_fraction * 2.0 * 4.0
    high_cost_decimal = (high_cost_bps / 10000.0) * avg_turnover_fraction * 2.0 * 4.0
    high_cost_pct_of_return = (high_cost_decimal / gross_return) * 100.0 if gross_return else float("nan")
    net_return = gross_return - high_cost_decimal

    summary = pd.DataFrame(
        {
            "Metric": [
                "Gross Return (%)",
                "Quarterly Turnover (Single-Counted)",
                f"Estimated Trading Costs ({int(low_cost_bps)} bps)",
                f"Estimated Trading Costs ({int(high_cost_bps)} bps)",
                f"Trading Costs ({int(high_cost_bps)} bps) as % of Return",
                "Net Return (%)",
            ],
            "U.S.": [
                gross_return * 100.0,
                avg_turnover_fraction * 100.0,
                low_cost_decimal * 100.0,
                high_cost_decimal * 100.0,
                high_cost_pct_of_return,
                net_return * 100.0,
            ],
        }
    )
    return summary, turnover_df


def render_trading_cost_table(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")

    display_df = summary.copy()
    display_df["U.S."] = display_df["U.S."].map(lambda x: f"{x:.2f}")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        colLoc="center",
        cellLoc="left",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_text_props(weight="bold", ha="center")
            cell.set_height(0.12)
        else:
            if col == 0:
                cell.set_text_props(weight="bold", ha="left")
            else:
                cell.set_text_props(ha="center")
            cell.set_height(0.13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
