# Financial Analyst Agent

A local AI assistant that answers questions about the annual reports (10-K) of six large US banks and about their yearly figures. Everything runs on your machine: the language model is served by Ollama, the embeddings run on CPU, and no data leaves the computer.

It has two parts that answer different kinds of questions:

| Question type | Example | How it is answered |
|---|---|---|
| Text in the 10-K | "How does Morgan Stanley describe its liquidity coverage ratio?" | **RAG**: hybrid search over the 10-K sections, reranking, and an answer with a checked quote and its source |
| Figures | "What was Truist's percentage change in total assets from 2019 to 2023?" | **Text-to-SQL**: the model writes a DuckDB query, which is validated and run read-only |

The agent that picks between the two parts is not built yet; today each part runs on its own.

## Data

- **Banks:** JPMorgan Chase (JPM), Bank of America (BAC), Citigroup (C), Goldman Sachs (GS), Morgan Stanley (MS) and Truist Financial (TFC).
- **Text:** the 3 most recent 10-K filings of each bank (fiscal years 2023–2025, 18 files) from SEC EDGAR, split into the *Risk Factors* and *Management's Discussion and Analysis* sections.
- **Figures:** `data/processed/finance.duckdb`, one table `financials` with 660 rows: 6 metrics (revenues, net income, assets, liabilities, stockholders' equity, deposits) for fiscal years 2006–2025, from the SEC XBRL "company facts" API. Amounts are in whole US dollars. See [`docs/schema.md`](docs/schema.md).

Nothing under `data/` is versioned. `make data` rebuilds it from scratch.

## How it works

**RAG** (`src/financial_analyst_agent/rag/`)
1. Sections are split into chunks of 1,000 characters (150 overlap). Each chunk starts with `company | FY | section | heading`, so citations show where a sentence comes from.
2. Chunks are embedded with `BAAI/bge-small-en-v1.5` and stored in Chroma (`data/chroma`).
3. A question is filtered by the bank and year it names, then searched with vectors and BM25 together (fused with RRF), and the top 30 candidates are reranked by `BAAI/bge-reranker-base`.
4. `llama3.2:3b` answers in JSON with a quote. The quote is checked against the cited chunk, and the answer is "I do not know" when the reports do not contain it.

**Text-to-SQL** (`src/financial_analyst_agent/sql/`)
1. The prompt includes the schema and the metric names (read at run time from `docs/schema.md` and `config/metrics.yaml`) plus 3 worked examples.
2. `llama3.2:3b` writes the query. `sql/validator.py` then checks it on the sqlglot syntax tree: exactly one SELECT, no DROP/DELETE/UPDATE/INSERT/ATTACH/COPY, no functions that read files, and `LIMIT` 100 at most.
3. The database is opened with `read_only=True` and `enable_external_access=False`.
4. If the query fails, the error is sent back to the model for a corrected query (at most 2 retries). Security rejections are never retried.

## Requirements

- Python 3.14 and [Poetry](https://python-poetry.org/) 2.x
- [Ollama](https://ollama.com/download) with `llama3.2:3b` (about 2 GB). `qwen2.5:3b` is also needed if you run the RAGAS evaluation.
- About 4 GB of free RAM for the evaluation runs (reranker plus model). A GPU is not needed: PyTorch is installed from the CPU index.

## Setup

```bash
poetry install
ollama pull llama3.2:3b
cp src/financial_analyst_agent/.env.example src/financial_analyst_agent/.env
```

In `src/financial_analyst_agent/.env`, set `SEC_USER_AGENT` to your name and email (the SEC requires it to download filings). Set `OLLAMA_BASE_URL` only if Ollama does not run on `http://localhost:11434`. The `.env` file is ignored by git.

Then build the data and the index:

```bash
make data                     # download 10-Ks, extract sections, load DuckDB (~1 min)
poetry run chunks             # split sections into chunks
poetry run index_chunks       # embed chunks into Chroma (~14 min on CPU)
```

## Usage

```bash
poetry run ask                                        # RAG: 5 sample questions with sources
poetry run python -m financial_analyst_agent.sql.tool # Text-to-SQL: 5 sample questions
poetry run python -m financial_analyst_agent.sql.validator "SELECT * FROM financials"
```

## Evaluation and checks

```bash
make test                                    # unit tests (pytest, no network or data needed)
poetry run python scripts/check_sql_safety.py # 40 safety cases for the SQL tool
poetry run rag_eval                          # retrieval hit@5 / MRR and answer grading
poetry run ragas_eval                        # RAGAS faithfulness and answer relevancy
poetry run python eval/run_figure_queries.py # rerun the SQL behind the figure answers
```

`eval/questions.csv` holds 40 evaluation questions: 25 about text and 15 about figures, whose expected answers come from hand-written SQL (`eval/figure_questions.sql`). These questions are kept out of prompts and test scripts so the measurements stay honest. Results are in [`docs/results.md`](docs/results.md), and the work log with failures and open issues is in [`docs/notes.md`](docs/notes.md).

## Project layout

```
src/financial_analyst_agent/
  config/      companies, RAG parameters, metric names, RAG eval questions
  ingest/      SEC download, HTML to text, 10-K sections, DuckDB load
  rag/         chunking, Chroma index, hybrid search, rerank, chain, evals
  sql/         DuckDB connection, prompt, chain, validator, LangChain tool
  llm.py       Ollama chat model (llama3.2:3b)
  embeddings.py
docs/          schema, results, work notes
eval/          evaluation questions and their SQL answers
scripts/       reusable checks (SQL safety)
tests/         pytest unit tests
```

## License

MIT. See [LICENSE](LICENSE).
