from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'output'

CONSERVATIVE_PATH = DATA_PROCESSED / 'conservative_portfolio_returns.parquet'
SPECULATIVE_PATH = DATA_PROCESSED / 'speculative_portfolio_returns.parquet'
MARKET_PANEL_PATH = DATA_PROCESSED / 'crsp_us_monthly_initial.parquet'
FACTOR_PANEL_PATH = DATA_PROCESSED / 'crsp_us_monthly_factors.parquet'
FF_PATH = DATA_PROCESSED / 'ff_factors_monthly.parquet'

METRICS_OUTPUT = OUTPUT_DIR / 'exhibit_5_factor_metrics.csv'
RETURNS_OUTPUT = OUTPUT_DIR / 'exhibit_5_factor_returns.csv'
CHART_OUTPUT = OUTPUT_DIR / 'exhibit_5_factor_metrics.png'
PRESENTATION_OUTPUT = OUTPUT_DIR / 'exhibit_5_factor_metrics_presentation.csv'

FACTOR_MAP = {
    'Formula': None,
    'Market': None,
    'Speculative': None,
    'Size': 'Lagged MKT Cap',
    'Momentum': 'Momentum',
    'LowVol': 'Volatility',
    'NPY': 'Net_Payout_Yield',
}

DISPLAY_NAME_MAP = {
    'Formula': 'Formula',
    'Market': 'Market',
    'Speculative': 'Speculative',
    'Size': 'Small',
    'Momentum': 'Momentum',
    'LowVol': 'Low Vol',
    'NPY': 'NPY',
}

ROW_ORDER = [
    'Return (simple) (%)',
    'Return (compounded) (%)',
    'Difference (Simple-Compounded) (%)',
    'Volatility (%)',
    'De-Risking Factor (%)',
    'Sharpe Ratio (simple)',
    'Return Same Risk (%)',
]

COLUMN_ORDER = ['Formula', 'Market', 'Size', 'Momentum', 'LowVol', 'NPY', 'Speculative']


def annualized_simple_return(returns: pd.Series) -> float:
    return float(returns.mean() * 12)


def annualized_compounded_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(12.0))


def de_risking_factor(conservative_returns: pd.Series, other_returns: pd.Series) -> float:
    observed_vol = annualized_volatility(other_returns)
    return float(annualized_volatility(conservative_returns) / observed_vol) if observed_vol else np.nan


def sharpe_ratio_simple(excess_returns: pd.Series, raw_returns: pd.Series) -> float:
    vol = annualized_volatility(raw_returns)
    return float(annualized_simple_return(excess_returns) / vol) if vol else np.nan


def return_at_same_risk(conservative_returns: pd.Series, other_returns: pd.Series, rf: pd.Series) -> float:
    scale = de_risking_factor(conservative_returns, other_returns)
    return float(scale * annualized_simple_return(other_returns) + (1.0 - scale) * annualized_simple_return(rf))


def ensure_yearmonth(frame: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    out = frame.copy()
    if 'YearMonth' not in out.columns:
        out['YearMonth'] = pd.to_datetime(out[date_col]).dt.to_period('M')
    elif not isinstance(out['YearMonth'].dtype, pd.PeriodDtype):
        out['YearMonth'] = pd.PeriodIndex(out['YearMonth'], freq='M')
    return out


def build_value_weighted_market_returns(panel: pd.DataFrame) -> pd.DataFrame:
    data = ensure_yearmonth(panel)
    data = data.copy()
    data['Total_Prev_MKT_Cap'] = data.groupby('YearMonth')['Lagged MKT Cap'].transform('sum')
    data['Weight'] = data['Lagged MKT Cap'] / data['Total_Prev_MKT_Cap']
    data['Weighted_Return'] = data['RET ADJ'] * data['Weight']
    market = (
        data.groupby('YearMonth', as_index=False)['Weighted_Return']
        .sum()
        .rename(columns={'Weighted_Return': 'Portfolio Returns'})
    )
    return market.iloc[1:].reset_index(drop=True)


def load_portfolio(path: Path) -> pd.DataFrame:
    return ensure_yearmonth(pd.read_parquet(path))


def filter_top_1000_by_mktcap(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.sort_values(by=['Date', 'MKT Cap'], ascending=[True, False])
        .groupby('Date', group_keys=False)
        .head(1000)
        .copy()
    )


def rank_top_100_by_factor(data: pd.DataFrame, factor_column: str) -> pd.DataFrame:
    ranked = data.copy()
    ascending = factor_column in {'MKT Cap', 'Lagged MKT Cap', 'Volatility'}
    ranked['Rank'] = ranked.groupby('Date')[factor_column].rank(ascending=ascending, method='first')
    return ranked.loc[ranked['Rank'] <= 100].drop(columns=['Rank']).copy()


def build_factor_portfolio_returns(data: pd.DataFrame, factor_column: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    sample = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
    rebalance_dates = sample.loc[sample['Date'].dt.month.isin([3, 6, 9, 12]), 'Date'].drop_duplicates().sort_values().reset_index(drop=True)

    portfolio_returns = []
    for i in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates.iloc[i]
        next_rebalance_date = rebalance_dates.iloc[i + 1]

        data_at_rebalance = sample.loc[sample['Date'] == rebalance_date].copy()
        top_1000 = filter_top_1000_by_mktcap(data_at_rebalance)
        top_100 = rank_top_100_by_factor(top_1000, factor_column)
        top_ids = top_100['PERMNO'].values

        holding_mask = (sample['Date'] > rebalance_date) & (sample['Date'] <= next_rebalance_date)
        holding_data = sample.loc[holding_mask & sample['PERMNO'].isin(top_ids), ['Date', 'PERMNO', 'RET ADJ']].copy()
        if holding_data.empty:
            continue

        returns_pivot = holding_data.pivot(index='Date', columns='PERMNO', values='RET ADJ')
        period_return = returns_pivot.mean(axis=1).to_frame(name='Portfolio Returns').reset_index()
        period_return['YearMonth'] = pd.to_datetime(period_return['Date']).dt.to_period('M')
        portfolio_returns.append(period_return[['YearMonth', 'Portfolio Returns']])

    if not portfolio_returns:
        return pd.DataFrame(columns=['YearMonth', 'Portfolio Returns'])

    return pd.concat(portfolio_returns, ignore_index=True)


def merge_with_rf(returns_df: pd.DataFrame, rf_df: pd.DataFrame) -> pd.DataFrame:
    merged = returns_df.merge(rf_df[['YearMonth', 'rf']], on='YearMonth', how='inner')
    merged['Excess Returns'] = merged['Portfolio Returns'] - merged['rf']
    return merged


def summarize_series(name: str, series_df: pd.DataFrame, cons_returns: pd.Series) -> dict[str, float | str]:
    returns = series_df['Portfolio Returns'].dropna()
    excess = series_df['Excess Returns'].dropna()
    rf = series_df['rf'].dropna()
    return {
        'Portfolio': name,
        'Return (simple) (%)': annualized_simple_return(returns) * 100.0,
        'Return (compounded) (%)': annualized_compounded_return(returns) * 100.0,
        'Difference (Simple-Compounded) (%)': (annualized_simple_return(returns) - annualized_compounded_return(returns)) * 100.0,
        'Volatility (%)': annualized_volatility(returns) * 100.0,
        'De-Risking Factor (%)': de_risking_factor(cons_returns, returns) * 100.0,
        'Sharpe Ratio (simple)': sharpe_ratio_simple(excess, returns),
        'Return Same Risk (%)': return_at_same_risk(cons_returns, returns, rf) * 100.0,
    }


def build_presentation_table(summary: pd.DataFrame) -> pd.DataFrame:
    present = summary.copy()
    present['DisplayPortfolio'] = present['Portfolio'].map(DISPLAY_NAME_MAP)
    present = present.set_index('DisplayPortfolio')[ROW_ORDER].T
    present = present.reindex(columns=[DISPLAY_NAME_MAP[name] for name in COLUMN_ORDER if name in DISPLAY_NAME_MAP])
    return present.round(1)


def save_chart(summary: pd.DataFrame) -> None:
    present = summary.copy()
    present['DisplayPortfolio'] = present['Portfolio'].map(DISPLAY_NAME_MAP)
    present['Portfolio'] = pd.Categorical(present['Portfolio'], categories=COLUMN_ORDER, ordered=True)
    present = present.sort_values('Portfolio')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#111111', '#7f7f7f', '#4d4d4d', '#2ca02c', '#ff7f0e', '#9467bd', '#c7c7c7']
    metrics = [
        ('Return (compounded) (%)', 'Compounded Return (%)'),
        ('Volatility (%)', 'Volatility (%)'),
        ('Sharpe Ratio (simple)', 'Sharpe Ratio'),
    ]

    for ax, (column, title) in zip(axes, metrics):
        ax.bar(present['DisplayPortfolio'], present[column], color=colors[: len(present)])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    fig.suptitle('Exhibit 5: Conservative Formula Versus Other Factors', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(CHART_OUTPUT, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conservative = load_portfolio(CONSERVATIVE_PATH)
    speculative = load_portfolio(SPECULATIVE_PATH)
    market_panel = pd.read_parquet(MARKET_PANEL_PATH)
    factor_panel = pd.read_parquet(FACTOR_PANEL_PATH)
    ff = ensure_yearmonth(pd.read_parquet(FF_PATH), date_col='Date')

    start_month = conservative['YearMonth'].min()
    end_month = conservative['YearMonth'].max()
    start_date = start_month.to_timestamp(how='end').normalize()
    end_date = end_month.to_timestamp(how='end').normalize()

    market = build_value_weighted_market_returns(market_panel)
    market = market.loc[(market['YearMonth'] >= start_month) & (market['YearMonth'] <= end_month)].copy()

    standalone_returns = {
        'Formula': conservative[['YearMonth', 'Portfolio Returns']].copy(),
        'Market': market[['YearMonth', 'Portfolio Returns']].copy(),
        'Speculative': speculative[['YearMonth', 'Portfolio Returns']].copy(),
    }

    for name, factor_column in FACTOR_MAP.items():
        if factor_column is None:
            continue
        standalone_returns[name] = build_factor_portfolio_returns(factor_panel, factor_column, start_date, end_date)

    returns_export = pd.concat(
        [frame.assign(Portfolio=name) for name, frame in standalone_returns.items()],
        ignore_index=True,
    )
    returns_export.to_csv(RETURNS_OUTPUT, index=False)

    rf_merged = {name: merge_with_rf(frame, ff) for name, frame in standalone_returns.items()}
    cons_returns = rf_merged['Formula']['Portfolio Returns']
    summary = pd.DataFrame([summarize_series(name, frame, cons_returns) for name, frame in rf_merged.items()])
    summary.to_csv(METRICS_OUTPUT, index=False)

    presentation = build_presentation_table(summary)
    presentation.to_csv(PRESENTATION_OUTPUT)
    save_chart(summary)

    print(f'Saved Exhibit 5 metrics to {METRICS_OUTPUT}')
    print(f'Saved Exhibit 5 return series to {RETURNS_OUTPUT}')
    print(f'Saved Exhibit 5 presentation table to {PRESENTATION_OUTPUT}')
    print(f'Saved Exhibit 5 chart to {CHART_OUTPUT}')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nPresentation Table:')
    print(presentation.to_string())
    print('\nNote: the paper exhibit uses Value, while this rebuild currently uses NPY because book-to-market is not in the factor panel.')


if __name__ == '__main__':
    main()

