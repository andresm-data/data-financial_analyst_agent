# Week 1: repeatable data pipeline

1. `make data` runs `run_all.py`, which runs download_10k, load_financials and manifest in order; it stops at the first failing step.
2. `data/processed/manifest.csv` lists each downloaded file with its source URL and download date (ISO 8601, UTC); dates already in the manifest are kept on reruns.
3. `rm -rf data && make data` rebuilt everything in ~42 s: 18 10-K files, 6 company facts, filings.csv, finance.duckdb (660 rows) and the manifest.
4. Failure: after the rebuild git was dirty, because `.gitkeep` and `filings.csv` were tracked and `manifest.csv`/`data/chroma` were not ignored; fixed by ignoring all of `data/`.
5. Pending: the run needs `SEC_USER_AGENT` in `src/financial_analyst_agent/.env` and network access to SEC; results change as new 10-Ks are filed.

# Week 2: local LLM and embeddings

## Ollama setup
1. Install Ollama from https://ollama.com/download (Linux: `curl -fsSL https://ollama.com/install.sh | sh`).
2. Start the server: `sudo systemctl start ollama` (or `ollama serve` in a terminal).
3. Pull the model: `ollama pull llama3.2:3b` (~2 GB, stored in Ollama's own directory, outside the repo).
4. Smoke test: `ollama run llama3.2:3b "In one sentence, what does a bank's net interest margin measure?"`.
5. `OLLAMA_BASE_URL` in `src/financial_analyst_agent/.env` overrides the default `http://localhost:11434`.

## Scripts
- `poetry run python -m financial_analyst_agent.llm` asks llama3.2:3b a question through LangChain (`ChatOllama`).
- `poetry run python -m financial_analyst_agent.embeddings` embeds 3 sentences with `BAAI/bge-small-en-v1.5` on CPU and prints 384 dimensions per vector.
- torch comes from the PyTorch CPU index (`pytorch-cpu` source in `pyproject.toml`), so no CUDA wheels are installed.
- The bge-small model (~130 MB) downloads on first run to `~/.cache/huggingface`. The "unauthenticated requests to the HF Hub" warning is harmless; set `HF_TOKEN` to silence it.
