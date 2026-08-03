# Conservative Formula Replication

This repo is my script-based replication of the Conservative Formula strategy from Blitz and van Vliet (2018), based on the earlier work of van Vliet and de Koning. The main goal was to move the project out of notebooks and into a reproducible pipeline: clean the monthly stock panel, build the factor inputs, form the conservative and speculative portfolios, and then recreate the paper's exhibits.

Most of the core logic lives in `src/conservative_formula/`, while the runnable workflow is in `scripts/`. Curated output tables/charts for the exhibits below are committed under `reports/`; `output/` is the scratch directory the scripts write to and is not tracked.

## Core pipeline

```powershell
python .\scripts\build_initial_panel.py
python .\scripts\build_factor_panel.py
python .\scripts\build_portfolios.py --start-date 1929-01-01 --end-date 2024-12-31
python .\scripts\evaluate_portfolios.py
```

Raw inputs live in `data/raw/` (requires a WRDS session, see `scripts/pull_wrds_monthly.py`), processed files are written to `data/processed/`.

## Exhibits

Each script defaults to the paper's original 1929-2016 sample (`--end-date 2016-12-31`). File names don't line up 1:1 with the paper's own numbering, since Exhibit 8 (international/midcap) is out of scope here:

| Script | Paper exhibit |
| --- | --- |
| `build_exhibit_2_summary.py` | Exhibit 2 (U.S. panel only) |
| `build_exhibit_3_growth.py` | Exhibit 3 |
| `build_exhibit_4_decades.py` | Exhibit 4 |
| `build_exhibit_5_factor_metrics.py` | Exhibit 5 |
| `build_exhibit_6_ff_comparison.py` | Exhibit 6 |
| `build_exhibit_7_factor_regressions.py` | Exhibit 7 |
| `build_exhibit_8_regime_alphas.py` | Exhibit 9 |
| `build_exhibit_9_trading_costs.py` | Exhibit 10 |

Two extensions beyond the base replication:
- `build_exhibit_5_factor_metrics.py --start-date 2017-01-01 --end-date 2023-12-31 --output-suffix _2017_2023` — Exhibit 5 recomputed for 2017-2023.
- `build_exhibit_7_factor_regressions.py` now produces three variants per panel: the paper's own conservative-minus-speculative spread (`cms`), the long-only conservative portfolio alone (`longonly`), and a beta-neutral long-short spread with both legs delevered/levered to a rolling 36-month beta of one (`betaneutral`).

## Notes

- The implementation is close to the paper, but it is still a replication, not a claim of exact reproduction.
- Results are close to the paper but not identical. The largest residual gaps concentrate in the pre-1963 period: the eligible universe is smaller than the paper's assumed 1,000-stock design and only crosses 1,000 stocks for the first time in March 1963, which makes factor selections more sensitive to data-construction differences in that era. Post-1963 results track the paper closely.
- A few choices follow the legacy project workflow where that matched the original results better than the paper text.
