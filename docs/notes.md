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

# Week 3: text-to-SQL tool

## Schema, metric names and the tool
1. `docs/schema.md` describes the only table, `financials` (660 rows: 6 banks, 6 metrics, fiscal years 2006–2025, whole US dollars), with its fallback XBRL tags, missing years, the 6 rows whose fiscal year ends on November 30 and the 2 rows with `value = 0`. `config/metrics.yaml` maps everyday words ("profit", "net income") to the 6 exact metric names.
2. The prompt (`sql/prompt.py`) reads the schema and the metric names at run time, so they are never copied into code, and adds 3 worked examples (direct lookup, ranking, percent change). None of them is in `eval/questions.csv`.
3. `sql/chain.py` runs prompt -> llama3.2:3b (temperature 0) -> SQL, and strips the ```` ```sql ```` fences the model often adds. `sql/tool.py` wraps it as the LangChain tool `text_to_sql`, which returns the SQL, the column names and the rows.
4. `sql/db.py` opens `finance.duckdb` with `read_only=True` and refuses to create a missing file, so a wrong path cannot leave an empty database behind.

## Safety checks, LIMIT and retries
1. `sql/validator.py` checks the sqlglot syntax tree (dialect `duckdb`), not the text, so a string `'DROP'` or a column `last_update` passes. It accepts exactly one SELECT (CTEs allowed, a trailing `;` ignored) and rejects several statements, any other statement, UNION, empty output and text it cannot parse.
2. DROP, DELETE, UPDATE, INSERT, ATTACH and COPY get their own message, also inside CTEs and subqueries. File-reading functions (`read_csv`, `read_parquet`, `read_json`, `read_text`, `read_blob`, `glob`, any `read_*`) and path-like table names (`FROM 'file.csv'`, which DuckDB reads straight from disk) are blocked too.
3. The executed SQL always has `LIMIT` 100 or less: added when missing, lowered when larger (also for `LIMIT 50%` or expressions), kept when smaller. `fetchmany(100)` stays as a second cap, and the connection now also sets `enable_external_access = False`, so DuckDB itself refuses `read_csv`, COPY and ATTACH even without the validator.
4. When DuckDB fails (or the SQL cannot be parsed), the question, the failed SQL and the error go back to the model for a corrected query, which is validated again: at most 2 retries (3 attempts). Security rejections and empty results (0 rows) are returned at once. Every attempt is logged (`financial_analyst_agent.sql.tool`, with `attempt`, `status`, `error_type` and `sql`) so failures can be grouped by cause; successful attempts are logged at INFO, so enable INFO to see them.
5. `poetry run python scripts/check_sql_safety.py` runs 40 invented cases (safe and dangerous queries, LIMIT rules, the connection without the validator, and retries with `FakeListChatModel`): 40/40 pass. The database checksum was the same before and after.

## Run on the 5 test questions (2026-11-20)
1. 4 of 5 answered on the first attempt: GS 2021 net income $21.635B, lowest 2022 liabilities TFC $494.7B, TFC assets +13.16% from 2019 to 2023, BAC 2014 equity-to-assets 0.1157. The 5 questions took ~3.7 min on CPU.
2. Failure: "Which had more total deposits in fiscal year 2017, JPMorgan Chase or Citigroup, and by how much?" failed twice with the same `BinderException` (a column alias used next to scalar subqueries), so the corrected query repeated the mistake. On the third attempt the model switched to `UNION ALL`, which the validator rejects, so the tool returned an error and no figures.
3. Pending: the repair prompt has the rules and the schema but not the 3 worked examples; adding the `MAX(CASE WHEN ...)` example could steer corrections away from scalar subqueries. Also decide whether to allow `UNION ALL` (today any UNION is rejected).
4. Pending: measure SQL accuracy on the 15 figure questions of `eval/questions.csv` (they were not used for tuning) and group the logged errors by `error_type`.

# Week 4: SQL accuracy on the 15 figure questions

## Measurement (sql_run1)
1. `eval/run_sql_eval.py <run>` runs the tool on the 15 figure questions (ids 26–40) and saves SQL, rows, error, attempts and time per question to `eval/results/<run>_raw.jsonl`; `eval/compare_sql.py <run>` scores it into `eval/results/<run>.csv`.
2. The expected values are the key columns of the hand-written queries in `eval/figure_questions.sql`, run read-only (not parsed from the CSV text). Numbers must be within 1%, banks must match as a set, and units are not normalized. The rule was fixed before looking at the answers.
3. sql_run1 (commit `17da7be`): **7/15 (46.7%)**. All 15 ran on the first attempt with no error; the 5 "which bank" questions and the 2 percent changes were right.

## Errors by cause (`eval/results/sql_run1_errors.csv`)
1. **wrong units: 5** (30, 35, 38, 39, 40). Right metric, year and logic, but the prompt asked for raw `value` in whole USD and the expected answers are in USD billions.
2. **other: 2** (31, 32). "Which was higher, and by how much?": the SQL returned both values and a signed difference but never said which bank was higher (the reference returns `higher` and `ABS(...)`).
3. **wrong column: 1** (33). Divided deposits by deposits (`pct_deposits = 1.0`) and never read `Assets`.
4. wrong year 0, bad SQL 0. Counting secondary causes, units appear in all 8 wrong answers.

## Change and sql_run2
1. One change, to the biggest group (commit `35af844`): prompt rule 4 now asks for amounts in USD billions (`value / 1e9`, alias ending in `_usd_bn`) and ratios in percent, without rounding; the 3 worked examples (none from the eval set) were updated to follow it, and the tool description says the same. `check_sql_safety.py` 40/40, tests 218/218.
2. sql_run2 (same questions, model, temperature 0, validator, runner and comparator): **9/15 (60.0%)**, +2 net. The goal of 12/15 (80%) was **not** reached.
3. Wrong -> right: 38, 39, 40 (units fixed). Right -> wrong: 37 (the new SQL failed with a `GROUP BY` error; the retry used `ANY_VALUE(company)` and an absolute difference instead of a percent, and answered JPM instead of TFC).
4. Still wrong, with a new failure: 30 divided by 1e9 and then multiplied back by 1e9; 35 used the `_usd_bn` alias without dividing; 31 and 32 now give the right difference in USD bn but still do not name the higher bank; 33 still ignores `Assets` and now returns `null`.
5. The units rule half-worked: simple lookups follow it, but in arithmetic expressions the 3B model applies `/ 1e9` inconsistently.

## Comparison rule fix
After sql_run2, ids 38–40 returned the right value in USD bn but were marked wrong because the bank-set check also ran when no bank is expected, and the model repeated the filtered `company` column. With approval, the bank set is now compared only when the expected answer names a bank (commit `6cd6578`). sql_run1 scores 7/15 under both versions (its CSV is byte-identical); sql_run2 goes from 6/15 to 9/15.

## Caveats
1. sql_run2 is measured on the same 15 questions used to diagnose the errors, so the improvement may be somewhat inflated, even though no eval question, answer or figure went into the prompt.
2. The comparison rule was corrected after seeing sql_run2. The fix only affects questions with no expected bank, and it does not change sql_run1.
3. With 15 questions, one question is 6.7 points; one run per configuration at temperature 0 does not show run-to-run variance.
4. Proposed held-out questions to check the change without that bias (not used for tuning; all have data in `finance.duckdb`; expected answers still to be written by hand):
   - What were Goldman Sachs' total liabilities at the end of fiscal year 2022?
   - Which had higher stockholders' equity in fiscal year 2021, Bank of America or Citigroup, and by how much?
   - What percentage of JPMorgan Chase's total assets were liabilities in fiscal year 2020?
   - By how many US dollars did Morgan Stanley's total assets change from fiscal year 2018 to 2022?
   - Which bank had the smallest percentage growth in total deposits between fiscal years 2016 and 2024?

## Pending (next iteration)
1. Add one invented worked example for "which was higher, and by how much?" that returns the higher bank and `ABS(...)` (targets 31 and 32; with no regressions it would reach 11/15).
2. Make the unit conversion consistent inside arithmetic (30, 35) and check whether 37 regressed because of the change or because of run-to-run instability.
3. Fix the ratio of two metrics (33), e.g. with an invented example that joins two metrics of the same bank.
