# Conservative Formula Replication

This repo is my script-based replication of the Conservative Formula strategy from van Vliet and de Koning. The main goal was to move the project out of notebooks and into a reproducible pipeline: clean the monthly stock panel, build the factor inputs, form the conservative and speculative portfolios, and then recreate the main evaluation tables and exhibit-style outputs.

Most of the core logic lives in `src/conservative_formula/`, while the runnable workflow is in `scripts/`.

## Core pipeline

```powershell
python .\scripts\build_initial_panel.py
python .\scripts\build_factor_panel.py
python .\scripts\build_portfolios.py --start-date 1929-01-01 --end-date 2024-12-31
python .\scripts\evaluate_portfolios.py
```

Raw inputs live in `data/raw/`, processed files are written to `data/processed/`, and charts/tables go to `output/`.

## Notes

- The implementation is close to the paper, but it is still a replication, not a claim of exact reproduction.
- A few choices follow the legacy project workflow where that matched the original results better than the paper text.
- The conservative side tracks the original project reasonably well. The speculative side is still the less perfect match.
- The repo also includes some extended-sample and exhibit work beyond the basic portfolio build.

