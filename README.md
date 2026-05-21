# Conservative Formula Replication

This repository contains a cleaned replication of the Conservative Formula strategy and related exhibits based on van Vliet and de Koning. It rebuilds the portfolio pipeline from CRSP-style monthly data, reproduces the main exhibit logic from the original project, and extends the sample beyond the original paper window.

The codebase is organized to keep raw-data preparation, portfolio construction, and exhibit generation separate so the workflow is easy to rerun and review.

## Repo structure

```text
src/conservative_formula/
    cleaning.py
    factors.py
    portfolio.py
    evaluation.py
    exhibits.py
    exhibit6.py
    exhibit7.py
    exhibit8.py
    exhibit9.py
scripts/
    build_initial_panel.py
    build_factor_panel.py
    build_portfolios.py
    evaluate_portfolios.py
    build_exhibit_3_growth.py
    build_exhibit_4_decades.py
    build_exhibit_5_factor_metrics.py
    prepare_exhibit_6_ff_data.py
    build_exhibit_6_ff_comparison.py
    prepare_exhibit_7_factor_data.py
    build_exhibit_7_factor_regressions.py
    prepare_exhibit_8_regime_data.py
    build_exhibit_8_regime_alphas.py
    build_exhibit_9_trading_costs.py
```

## Main workflow

Run the core portfolio pipeline:

```powershell
python .\scripts\build_initial_panel.py
python .\scripts\build_factor_panel.py
python .\scripts\build_portfolios.py --start-date 1929-01-01 --end-date 2024-12-31
python .\scripts\evaluate_portfolios.py
```

Build exhibits:

```powershell
python .\scripts\build_exhibit_3_growth.py
python .\scripts\build_exhibit_4_decades.py
python .\scripts\build_exhibit_5_factor_metrics.py
python .\scripts\prepare_exhibit_6_ff_data.py
python .\scripts\build_exhibit_6_ff_comparison.py
python .\scripts\prepare_exhibit_7_factor_data.py
python .\scripts\build_exhibit_7_factor_regressions.py
python .\scripts\prepare_exhibit_8_regime_data.py
python .\scripts\build_exhibit_8_regime_alphas.py
python .\scripts\build_exhibit_9_trading_costs.py
```

Generated outputs are written to `output/` and processed intermediate datasets are written to `data/processed/`.

## Data inputs

The project expects raw data files in `data/raw/`. These include:

- CRSP-style monthly stock panel data
- Kenneth French monthly factor files
- Fama-French sorted portfolio files for Exhibit 6
- Hou-Xue-Zhang and AQR factor files for Exhibit 7
- FRED macro series for Exhibit 8

Raw and processed data are ignored in Git because they are large and reproducible from the original downloads.

## Replication notes

- The implementation is structurally close to the paper, but not every detail matches perfectly.
- A legacy notebook-style net payout yield definition reproduces the original project results more closely than the paper-text version.
- The conservative portfolio tracks the original project reasonably well.
- The speculative portfolio remains somewhat stronger than in the paper, so that is an open replication difference.
- The `Small` benchmark in the factor comparison exhibit is a custom stock-sorted implementation and should not be treated as a perfect paper benchmark.
- The repository includes extended-sample analysis beyond the original paper window.

## What this repo is meant to show

- a reproducible research pipeline
- paper-oriented portfolio construction
- script-based exhibit generation
- clear separation between raw data prep and final outputs

