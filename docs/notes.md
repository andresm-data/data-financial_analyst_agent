# Week 1: repeatable data pipeline

1. `make data` runs `run_all.py`, which runs download_10k, load_financials and manifest in order; it stops at the first failing step.
2. `data/processed/manifest.csv` lists each downloaded file with its source URL and download date (ISO 8601, UTC); dates already in the manifest are kept on reruns.
3. `rm -rf data && make data` rebuilt everything in ~42 s: 18 10-K files, 6 company facts, filings.csv, finance.duckdb (660 rows) and the manifest.
4. Failure: after the rebuild git was dirty, because `.gitkeep` and `filings.csv` were tracked and `manifest.csv`/`data/chroma` were not ignored; fixed by ignoring all of `data/`.
5. Pending: the run needs `SEC_USER_AGENT` in `src/financial_analyst_agent/.env` and network access to SEC; results change as new 10-Ks are filed.
