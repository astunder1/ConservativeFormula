# Data

This project does not commit raw WRDS/CRSP data to Git.

## Layout

```text
data/
    README.md
    raw/
    processed/
```

## Policy

- `raw/` is for local WRDS extracts and should stay out of version control.
- `processed/` is for locally generated research panels and should also stay out of version control unless a tiny synthetic sample is created for tests.
- The repository should contain extraction code, transformation code, and documentation, but not proprietary source data.

## Main replication target

The Conservative Formula replication needs a monthly U.S. stock panel that supports:

- universe selection among the largest 1,000 U.S. stocks
- 36-month volatility estimation
- 12-1 momentum
- dividend yield
- net payout yield using shares outstanding
- delisting-adjusted returns
- quarterly rebalancing

## Legacy notebook columns to reproduce

The old notebook expected columns with these names or equivalents:

- `PERMNO`
- `Date`
- `RET`
- `RETX`
- `DLRET`
- `DLRETX`
- `DLSTCD`
- `ALTPRC`
- `SHROUT`
- `DIVAMT`
- `EXCHCD`
- `SHRCD`

## Extraction

`scripts/pull_wrds_monthly.py` builds the monthly panel with the fields above and maps raw column names into the schema used by `src/conservative_formula/`.
