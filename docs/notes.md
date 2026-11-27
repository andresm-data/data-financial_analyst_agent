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
6. Errors of sql_run2 by cause (`eval/results/sql_run2_errors.csv`): other 3 (31 and 32 do not name the higher bank; 37 aggregates all banks after a failed `GROUP BY`), wrong units 2 (30, 35), wrong column 1 (33). The biggest group is now "which is higher" (31, 32).

## Second iteration (sql_run3)
1. One change on top of sql_run2, approved (commit `184b6ec`): a 4th invented worked example, "Which had larger total liabilities in fiscal year 2019, Truist Financial or Goldman Sachs, and by how much?", whose SQL self-joins the two banks and returns `higher` and `ABS(...) / 1e9`. It is not in `eval/questions.csv` nor in `eval/control_questions.sql`, and it returns 1 row. `check_sql_safety.py` 40/40, tests 219/219.
2. sql_run3 (same questions, model, temperature 0, validator, runner and comparator): **10/15 (66.7%)**, +1 net over sql_run2. The goal of 12/15 (80%) was **not** reached.
3. Wrong -> right: 31 and 32 (now name the higher bank with the absolute difference, the target of the change), and 30 and 35 (the self-join style also fixed their units). Right -> wrong: 29 (ranking without ORDER BY / LIMIT 1), 34 and 36 (percent change written as `(new - old) * 100` with no division by the old value, and 36 has no year filter). 33 now reads Assets but divides by 1e9 inside the ratio; 37 is still wrong.
4. Errors by cause (`eval/results/sql_run3_errors.csv`): other 3 (29, 34, 37), wrong units 1 (33), wrong year 1 (36). Three of the five (33, 34, 36) put `/ 1e9` inside a ratio or percent change; 37 also needs a percent change.
5. Lesson: llama3.2:3b copies the shape of the newest example to other question types (the self-join spread to percent changes) and a longer prompt changed an unrelated ranking question (29). Each example fixes its pattern and can break others, so gains on 15 questions are fragile.

## Third iteration (sql_run4)
1. One change on top of sql_run3, approved (commit `5cd4280`): prompt rule 4 now also says to compute ratios and percent changes on the raw `value` (`100 * (new - old) / old`, `100 * a / b`) and to divide by 1e9 only the returned amounts. `check_sql_safety.py` 40/40, tests 219/219.
2. sql_run4 (same questions, model, temperature 0, validator, runner and comparator): **12/15 (80.0%)**, the goal, +2 net over sql_run3.
3. Wrong -> right: 33 (now divides deposits in USD bn by assets in USD bn, so the 1e9 cancels: 59.17%) and 29 (the ranking has `ORDER BY value ASC LIMIT 1` again, although the change did not target rankings). No right -> wrong.
4. Still wrong (`eval/results/sql_run4_errors.csv`): other 2 (34: `100 * (new - old)` in USD bn without dividing by the old value; 37: no self-join between 2015 and 2025) and wrong column 1 (36: no metric filter, 6 rows). The rule's percent-change formula was ignored in 34 and 36, which were its main targets.

## Honest reading of the 80%
1. The goal was reached on the same 15 questions used to diagnose the errors and to choose 3 successive changes (units, the "which is higher" example, the ratio rule). The 80% is therefore an optimistic estimate, more than after a single change.
2. Of the 2 questions gained in sql_run4, only 33 is explained by the change; 29 came back without being targeted, as it had broken in sql_run3 without being targeted. With llama3.2:3b any prompt change moves unrelated questions, and one question is 6.7 points, so 80% could be 73% or 87% with a different but equally valid prompt.
3. Percent change between two years is still the weakest pattern (34, 36, 37 all fail it), even with the formula in the rules.
4. Before trusting the result: write the expected answers of the 5 control questions (`eval/control_questions.sql`, never used for tuning), score the final prompt on them, and run sql_run4 again to check it is stable.

## Control questions (control_run4)
1. With approval, the 5 control questions were added to `eval/questions.csv` as ids 41–45 with `type = control`, so the 15 `figure` questions and the 25 `text` ones did not change and sql_run1–4 still score the same. Their hand-written SQL is in `eval/control_questions.sql`, the key columns were fixed in `eval/compare_sql.py` before running them, and the runner takes `--type control`.
2. The final prompt (sql_run4) on the control questions: **2/5 (40.0%)**, against 12/15 (80.0%) on the 15 tuning questions.
3. Right: 41 (direct lookup) and 42 ("which is higher", the pattern of the newest example). Wrong (`eval/results/control_run4_errors.csv`): 43 wrong column (never reads Assets, like 33), 44 wrong year (joins each year with the next one instead of 2018 with 2022, and whole USD), 45 other (no self-join between the two years, like 37).
4. Conclusion: the 80% does not generalize. The prompt learned the patterns that have a worked example (lookup, ranking, "which is higher") but not the ones without one: a ratio of two metrics, a change between two years and a growth ranking. The 15 questions overstate the real accuracy, as the caveat warned.

## Fourth iteration (sql_run5, control_run5)
1. One change on top of sql_run4, approved (commit `450891f`): a 5th invented worked example, "How much did Bank of America's total liabilities change from fiscal year 2012 to 2017, in US dollars and in percent?", a self-join of one bank's two years on `company` and `metric` that returns the change in USD bn and `100 * (new - old) / old`. Not in the 15 nor in the control questions; it returns 1 row. `check_sql_safety.py` 40/40, tests 220/220.
2. sql_run5: **10/15 (66.7%)**, -2 against sql_run4. control_run5: **1/5 (20.0%)**, -1 against control_run4. Worse on both sets.
3. Wrong -> right: 36 (now 25.92%). Right -> wrong: 30, 32 and 33 on the 15, and 42 on the control set. 34 now has the right formula and the right first row, but no `metric` filter (6 rows).
4. Why: the model copied the example's join `ON y1.company = y0.company AND y1.metric = y0.metric` into comparisons of two banks (30, 32, 42) and ratios of two metrics (33, 43), where the join must be on `fiscal_year` or `company` only, so they return 0 rows or the wrong rows. Errors (`eval/results/sql_run5_errors.csv`, `control_run5_errors.csv`): wrong column 4 + 2, other 1 + 2.
5. The control questions guided this change (their failures 44 and 45 motivated it), so control_run5 is no longer a held-out measurement; a new control set would be needed to measure generalization from here on.
6. Lesson: with llama3.2:3b, each worked example is copied into question types it was not meant for. Four prompt changes moved the score on the 15 questions between 9 and 12, while the control set stayed at 1–2 of 5. sql_run4 is the best prompt measured.

## Comparison rule fix
After sql_run2, ids 38–40 returned the right value in USD bn but were marked wrong because the bank-set check also ran when no bank is expected, and the model repeated the filtered `company` column. With approval, the bank set is now compared only when the expected answer names a bank (commit `6cd6578`). sql_run1 scores 7/15 under both versions (its CSV is byte-identical); sql_run2 goes from 6/15 to 9/15.

## Caveats
1. sql_run2 is measured on the same 15 questions used to diagnose the errors, so the improvement may be somewhat inflated, even though no eval question, answer or figure went into the prompt.
2. The comparison rule was corrected after seeing sql_run2. The fix only affects questions with no expected bank, and it does not change sql_run1.
3. With 15 questions, one question is 6.7 points; one run per configuration at temperature 0 does not show run-to-run variance.
4. Proposed control questions to check the change without that bias (not used for tuning, not in `eval/questions.csv`; hand-written SQL in `eval/control_questions.sql`, each returns 1 row; to be added to the eval set only after approval):
   - What were Goldman Sachs' total liabilities at the end of fiscal year 2022?
   - Which had higher stockholders' equity in fiscal year 2021, Bank of America or Citigroup, and by how much?
   - What percentage of JPMorgan Chase's total assets were liabilities in fiscal year 2020?
   - By how many US dollars did Morgan Stanley's total assets change from fiscal year 2018 to 2022?
   - Which bank had the smallest percentage growth in total deposits between fiscal years 2016 and 2024?

## Pending
1. Done: `450891f` (the 5th example) was reverted with a new commit, so the final prompt is the sql_run4 one again (`src/` identical to `5cd4280`). Its expected scores are 12/15 and 2/5; a repeat run would also check stability.
2. More examples are not converging with a 3B model. Options: build the SQL from templates per question type (lookup, ranking, two-bank comparison, two-year change, ratio of two metrics) with the model only extracting bank, metric and years; or try a larger local model. Either change needs a new held-out set to be measured fairly.
3. The tool returns rows as dicts, so two columns with the same alias overwrite each other (seen in 31 of sql_run2).

# Week 5: agent classifier

## Route model and classifier node (2026-11-26)
1. `src/financial_analyst_agent/agent/schemas.py`: `RouteDecision` (Pydantic 2) with `route` as `Literal["text", "numbers", "both", "out_of_scope"]` and a one-sentence `reason`. `ROUTES` is the shared tuple for `add_conditional_edges`; a test checks it matches `docs/agent_graph.md`.
2. `src/financial_analyst_agent/agent/classifier.py`: `classify(state)` reads `question` and returns `route` and `route_reason`. The prompt defines the 4 routes, reads the 6 metrics from `config/metrics.yaml` at run time, says not to judge whether the bank or year exists, and has 8 invented examples (2 per route).
3. `llm.with_structured_output(RouteDecision)` with llama3.2:3b at temperature 0. `json_schema` (the default), `function_calling` and `json_mode` all parsed on a 3-question check; `json_schema` is kept.
4. No example is in `eval/questions.csv`: compared against all 45 rows by normalized text and `difflib` (fail above 0.8); highest ratio 0.481. Two `text` examples were replaced (ratios 0.545 and 0.555) because they shared a topic with an eval question. Checked by `tests/test_classifier.py`.

## Test on 10 questions (classifier_run1)
1. Rule fixed before running: expected route from `type` (`text` -> `text`, `figure` -> `numbers`), `control` rows excluded, `random.Random(42)`, 5 sampled from the sorted `text` ids then 5 from the sorted `figure` ids. Selected: 21, 4, 1, 24, 9 and 29, 40, 28, 37, 27.
2. Result: **10/10**, no parse errors (`eval/results/classifier_run1.csv`). Run metadata in `classifier_run1_meta.json`: llama3.2:3b, temperature 0, `json_schema`, 2026-11-26T14:16:53Z, code at `641a87f`.
3. Separate check, not counted (`eval/results/classifier_extra_check.csv`, 6 invented questions, not in the eval set): **5/6**. Both `both`, both `out_of_scope` and the narrative-phrased figure question were right. "What was JPMorgan Chase's CET1 capital ratio in 2024?" (expected `text`) went to `out_of_scope`; the reason said 2024 is not an available year, which breaks the rule not to judge whether the data exist.

## Caveats
1. 10 questions only cover `text` and `numbers`; `both` and `out_of_scope` were checked on 6 invented questions only. One run at temperature 0 does not show run-to-run variance.
2. `eval/questions.csv` labels figure questions as `figure`, not `numbers`; the mapping is a fixed rule in `eval/run_classifier_eval.py`.

## Pending
1. Proposed single change, not applied: a `text` example about a metric not in the table (for example a capital ratio) for a bank and year outside the data, to fix the CET1 case.
2. Decide whether the 5 `control` questions count in the routing measurement on 1 Dec (40 or 45 questions).

## Agent graph with RAG, SQL and answer nodes (2026-11-27)
1. Branch from `main` (the classifier branch was already merged in PR #14). Pre-check: `classify`, `ROUTES` and the planned state were in place; classifier_run1 10/10; `config/rag.yaml` uses the 1000-character index, as `docs/results.md` concludes; `check_sql_safety.py` 40/40; Ollama 0.17.7 with llama3.2:3b. The SQL tool already returned `sql`, `rows` and `error` separately, so no refactor was needed.
2. State: `AgentState` in `agent/state.py` (`TypedDict`, `total=False`, the 8 fields of `docs/agent_graph.md`). Nodes: `retrieve_text` and `query_sql` in `agent/tools.py` reuse `rag.chain.default_retriever` (k=5) and `sql.tool.answer_with_sql` without changes; `answer` in `agent/answer.py`; `refuse` in `agent/graph.py` is **provisional until 30 Nov** (fixed message).
3. Graph: `agent/graph.py`, `build_graph()` compiles once. `classify` has a conditional edge over `ROUTES` (checked to cover every route); `both` runs `retrieve_text` then `query_sql` in sequence. The edges out of `retrieve_text` use route names as keys, so the drawing matches the design except one label (see `docs/agent_graph.md`).
4. `answer`: one model call with the RAG chain's strict `SYSTEM_RULES` (unchanged) plus 4 rules (cite sources, US dollars, answer both parts, the table figure wins over the text). With no context, or only an error, it answers with a fixed message and does not call the model. Added in code, because the 3B model does not follow the citation and precedence rules reliably: a `Sources:` block from the state, and in mixed answers the table rows as reference values.
5. Context window: Ollama loads the model with `num_ctx` 4096 (server default; `get_llm` does not set it). Only the first 20 SQL rows reach the prompt (`MAX_PROMPT_ROWS`), and the answer says when rows were left out. Worst case measured (5 chunks + 20 of 100 rows): 2443 tokens, 60%. The node logs prompt tokens and warns above 80%.

## End-to-end test (scripts/try_agent_graph.py)
3 invented questions, not in `eval/questions.csv` (the script checks it), so the routing measurement on 1 Dec and the full test on 7 Dec are not inflated.

| Type | Question | Route | Prompt tokens | Time |
|---|---|---|---|---|
| text | Climate change risks in Goldman Sachs' 2024 10-K | text | 1045 | 77 s |
| numbers | Morgan Stanley total deposits, FY2022 | numbers | 368 | 124 s |
| both | JPMorgan Chase net income FY2024 and what drove it | both | 1828 | 219 s |
| not counted | Should I buy Citigroup stock before its next earnings report? | out_of_scope -> `refuse` | – | 29 s |

1. Numbers: "USD 356.646 billion" matches the SQL run by hand in read-only mode (356,646,000,000 USD).
2. Text: all 5 chunks are Goldman Sachs FY2024 (`GS_2025-02-27_gs-20241231.htm`).
3. Mixed: both parts present. The model writes "$58.5 billion", copied from the 10-K text; the table value 58.471 is added in code as the reference. The percentages (18%, 12%, 23%, 4%) are in the retrieved excerpts. No figure without support.

## Caveats
1. The `answer` rules were adjusted on the same 3 test questions: the first run invented the citation "(MS, 2022, Financial Highlights)" and rounded a figure. Rule 9 (the table wins) did not change the mixed answer, and the text answer went from 4 sentences to 1 (still correct and supported).
2. Inline citations from the model omit company and year; the code-generated `Sources:` block is the reliable one.
3. The SQL of the mixed question has a misleading comment written by the model ("the 10-K filing is not available"); harmless, left for 30 Nov because the SQL tool was not to be changed today.
4. Latency on CPU: 77–219 s per question; the mixed path makes 3 model calls plus the rerank.

## Pending
1. 30 Nov: final `refuse` node and error handling (an unparseable classifier output still raises), and strip comments from generated SQL.
2. Watch answer length on 7 Dec: if text answers stay at one sentence, compare with and without rule 9 on a held-out set, not on these 3 questions.
