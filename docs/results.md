# Results

## Chunk size: 1000 vs 500

RAGAS on the same 12 text questions for both sizes, scored by `qwen2.5:3b` (temperature 0) on 2026-11-18. Generator `llama3.2:3b`, hybrid search + rerank, k=5. Per-question scores: `eval/results/chunk_1000.csv` and `eval/results/chunk_500.csv`.

| Chunk size (overlap) | Chunks in index | Faithfulness | Answer relevancy | "I do not know" |
|---|---|---|---|---|
| 1000 (150) | 9,629 | **0.592** | 0.643 | 2/12 |
| 500 (75) | 19,157 | 0.507 | **0.710** | 1/12 |

By question group (faithfulness / answer relevancy):

| Group | 1000 | 500 |
|---|---|---|
| Risk factors (ids 1, 2, 5, 6, 10, 11) | 0.583 / 0.635 | 0.667 / 0.746 |
| MD&A results (ids 14, 15, 16, 20, 21, 22) | 0.600 / 0.649 | 0.347 / 0.680 |

**Conclusion:** neither size wins clearly. 1000 is more faithful to the retrieved context (+0.085), while 500 gives more relevant answers (+0.067) and abstains less. The faithfulness drop with 500 comes from the MD&A results questions. With 12 questions, one question moves the average by about 0.08, so these differences are within noise. Faithfulness matters most for a financial agent, so 1000 stays the reference until a run on all 25 questions confirms the result.

Notes:
- Subset: 12 of the 25 text questions (6 risk, 6 results, two or three per bank) to cut the run from ~2 h to ~70 min on CPU. Run with `poetry run ragas_eval --ids 1,2,5,6,10,11,14,15,16,20,21,22`.
- Question 2 has no answer relevancy score (NaN) in both runs: the answer was "I do not know" and the judge returned no valid score. Answer relevancy only uses the question and the answer, so the judge got the same input for both sizes. The answer relevancy averages are over the 11 scored questions. Counting it as 0 would give 0.589 (1000) and 0.651 (500).

## SQL accuracy on the 15 figure questions

The text-to-SQL tool (`sql/tool.py`: sqlglot validator, LIMIT 100, up to 2 retries) answers the 15 figure questions of `eval/questions.csv` (ids 26–40). Run with `poetry run python eval/run_sql_eval.py <run>` and scored with `poetry run python eval/compare_sql.py <run>`. Per-question results: `eval/results/<run>.csv` (raw output in `<run>_raw.jsonl`).

Comparison rule (fixed before looking at the answers): the expected values are the key columns of the hand-written queries in `eval/figure_questions.sql` (the bank for "which bank" questions, the difference, percent or amount otherwise). An answer is correct if it has no error, the same number of rows as the reference, the same set of banks (ticker or name, case- and space-insensitive) when the expected answer names a bank, and every expected number within 1% relative difference. Correction made after sql_run2 and approved: the bank set was first also compared when no bank is expected, so a `company` column repeating the filtered bank (ids 38–40) counted as wrong; sql_run1 scores the same under both versions. Units and scale are not normalized: an amount in whole USD against an expected value in USD billions, or a fraction against a percent, is wrong.

| Run | Model | Temperature | Date | Commit | Correct (of 15) | Accuracy |
|---|---|---|---|---|---|---|
| sql_run1 | llama3.2:3b | 0 | 2026-11-24 | `17da7be` | 7 | 46.7% |
