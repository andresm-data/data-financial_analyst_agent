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

## RAG chain with citations
1. Built the RAG chain with LCEL (`rag/chain.py`): Chroma retriever with k=5 and the bank/year filter, strict prompt (`rag/prompt.py`), `ChatOllama` llama3.2:3b and `StrOutputParser`; it returns the answer and the source chunks.
2. `poetry run ask` runs 5 test questions (~4.5 min on CPU) and prints company, year and section for each source; Goldman Sachs 2025 and Bank of America 2023 got cited answers, the Apple question got "I do not know".
3. Failure: llama3.2:3b often wrote "The context does not provide..." and then talked about another year; answers that start like that are now turned into exactly "I do not know".
4. Failure: adding more rules to the prompt made it worse (it gave Citi's 5.8% leverage ratio as the CET1 ratio), so the prompt stayed short; the model also said BAC deposits "increased" from $1.93T to $1.92T.
5. Pending: Citi's 2024 CET1 ratio (13.63%) is in the reports, but the retriever returned the next chunk of the same table without that row, and JPM 2024 net interest income drivers were not retrieved; retrieval of tables needs work (hybrid search or reranking).

## Retrieval evaluation, hybrid search and reranking
1. `poetry run rag_eval` scores 12 questions (`config/eval_questions.yaml`: 9 answerable, 3 not in the 10-Ks) and saves each run to `data/eval/`; `--retrieval-only` skips the LLM (hit@5 and MRR in seconds).
2. Baseline (vector search, k=5): hit@5 0.67, MRR 0.45; answers 8 correct, 2 missed, 2 wrong, 0 invented. A reranker alone did not help, because the right chunks were not even in the top 30 vector results.
3. Hybrid search (vectors + BM25, fused with RRF, 30 candidates) reranked by `BAAI/bge-reranker-base`: hit@5 0.89, MRR 0.76; 10 correct, 1 missed, 1 wrong, 0 invented. Fixed Citi CET1 (13.63%) and BAC deposits ("decreased $6.5 billion"). Costs ~10 s more per question and ~1 GB RAM; the full eval needs ~3.5 GB free (one run was killed for low memory).
4. Answers now come as JSON with a quote that is checked against the cited chunk (`rag/grounding.py`), and only that chunk is cited: 10 correct, 0 missed, 2 wrong, 0 invented. Failure: for JPM 2024 NII drivers it now answers with the 2025 outlook instead of "I do not know", and for JPM total assets it quotes the Asset & Wealth Management segment table ($255,385 million) instead of the firm's $4.0 trillion.
5. Not done: a relevance threshold on the reranker score does not work here (Truist "bitcoins", not in the reports, scores 0.957; BAC deposits, answerable, 0.541). The BGE query prefix made retrieval worse (Citi CET1 dropped from rank 11 to 20). Pending: segment/table titles in the chunks (needs reindexing) so the model can tell firm-wide figures from segment ones.

## Section headings in chunks
1. `rag/headings.py` finds the headings in each 10-K section (ALL CAPS for JPM and Citi, Title Case for GS, BAC, MS and Truist) and skips page headers and table column headers; 647 of 9,629 chunks have no heading (mostly Risk Factors, whose headings are full sentences).
2. Each chunk now starts with `company | FY | section | heading` and citations show the heading, e.g. `JPMorgan Chase | FY2024 | Management's Discussion and Analysis > CORPORATE`; the prefix is added after splitting, so chunk ids did not change. Reindexing took ~14 min on CPU.
3. Result: 11 correct, 1 missed, 0 wrong, 0 invented (before: 10 / 0 / 2 / 0). Retrieval hit@5 fell from 0.89 to 0.67 and MRR from 0.76 to 0.59, partly because other chunks also hold the answer (Citi CET1 13.6% in `C_2024_mdna_0011`).
4. Failure: JPM total assets went from a wrong answer (segment table) to "I do not know", because `JPM_2024_mdna_0002` ($4.0 trillion) dropped out of the top 5.
5. Caveat: the JPM 2024 NII answer is graded correct but quotes the Corporate segment ($9.8 billion NII); the drivers match the firm-wide ones, and the citation now shows "CORPORATE", so a reader can spot it.

## Single-word headings and the heading of the quote
1. One-word Title Case lines ("Deposits", "Overview", "Liquidity", "Market") now count as headings when a paragraph follows them; otherwise they are usually table header pieces ("Average", "Total"). Chunks without a heading fell from 647 to 60, mostly in Risk Factors.
2. A chunk can span several subsections, so the cited source now takes the last heading before the quoted sentence: BAC 2023 deposits cites "> Deposits" (was "All Other Assets") and Citi 2024 CET1 cites "> EXECUTIVE SUMMARY > Capital" (was "Cost of Credit").
3. Only the 1,795 chunks whose heading changed were re-embedded (~2 min instead of ~14 min).
4. Eval: 10 correct, 1 missed, 1 wrong, 0 invented; hit@5 0.67, MRR 0.54. The "wrong" one is GS 2025 AWM risks: it quotes the same chunk as before but another bullet (market uncertainty), a real risk that lacks the expected keywords; the grader's keyword match is too strict for open questions.
5. Pending: JPM total assets still answers "I do not know" because the $4.0 trillion chunk is not retrieved, and llama3.2:3b picks different sentences between runs, so open questions need a better grader (e.g. several accepted answers or an LLM judge).
