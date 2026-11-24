"""Run the text-to-SQL tool on the 15 figure questions of eval/questions.csv.

For each question it saves the id, the question, the final executed SQL, the
columns and rows returned, the error (if any), the number of attempts (with
retries) and the time taken. A question that fails does not stop the rest.

The output goes to `eval/results/<run_name>_raw.jsonl` (one JSON object per
question) plus a first line with the run metadata (model, temperature, date,
commit). Score it with `eval/compare_sql.py`.

Only a status line per question is printed, not the rows, so the answers can
be compared with a rule fixed beforehand.

Usage:
    poetry run python eval/run_sql_eval.py sql_run1
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from financial_analyst_agent.llm import MODEL_NAME
from financial_analyst_agent.sql.tool import answer_with_sql

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "eval" / "questions.csv"
RESULTS_DIR = ROOT / "eval" / "results"
FIGURE_TYPE = "figure"  # type of the 15 figure questions in questions.csv
TEMPERATURE = 0.0       # the SQL chain is built with get_llm(temperature=0.0)


def figure_questions(path: Path = CSV_PATH) -> list[dict[str, str]]:
    """The rows of questions.csv with type = figure, in file order."""
    with path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["type"] == FIGURE_TYPE]


def git_commit() -> str:
    """Short hash of HEAD, with `-dirty` if the working tree has changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def run_question(row: dict[str, str], answer=answer_with_sql) -> dict:
    """Run one question; any exception is recorded as the error."""
    start = time.perf_counter()
    try:
        result = answer(row["question"])
    except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
        result = {"sql": "", "columns": [], "rows": [], "attempts": 0,
                  "error": f"{type(exc).__name__}: {exc}"}
    return {
        "id": row["id"],
        "question": row["question"],
        "sql": result.get("sql", ""),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "error": result.get("error"),
        "attempts": result.get("attempts", 0),
        "seconds": round(time.perf_counter() - start, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_name", help="e.g. sql_run1, sql_run2")
    args = parser.parse_args()

    rows = figure_questions()
    out_path = RESULTS_DIR / f"{args.run_name}_raw.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"run": args.run_name, "model": MODEL_NAME, "temperature": TEMPERATURE,
            "date": date.today().isoformat(), "commit": git_commit(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_questions": len(rows)}

    with out_path.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"meta": meta}) + "\n")
        for row in rows:
            record = run_question(row)
            out.write(json.dumps(record, default=str) + "\n")
            out.flush()
            status = "error" if record["error"] else f"{len(record['rows'])} rows"
            print(f"id {record['id']}: {status}, attempts={record['attempts']}, "
                  f"{record['seconds']}s", flush=True)

    print(f"Saved {len(rows)} questions to {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
