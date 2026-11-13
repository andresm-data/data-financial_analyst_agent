"""Run the hand-written SQL in eval/figure_questions.sql and print each result.

Use it to regenerate or audit the expected answers of the figure questions in
eval/questions.csv. The database is opened read-only.

Usage:
    poetry run python eval/run_figure_queries.py
"""

import csv
import re
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "finance.duckdb"
SQL_PATH = ROOT / "eval" / "figure_questions.sql"
CSV_PATH = ROOT / "eval" / "questions.csv"


def parse_queries(text: str) -> dict[str, str]:
    """Split the SQL file into {question id: query} using the `-- id: N` markers."""
    queries = {}
    for block in re.split(r"(?m)^-- id: ", text)[1:]:
        qid, _, body = block.partition("\n")
        sql = "\n".join(
            line for line in body.splitlines() if not line.startswith("--")
        ).strip()
        queries[qid.strip()] = sql
    return queries


def main() -> int:
    queries = parse_queries(SQL_PATH.read_text())
    con = duckdb.connect(str(DB_PATH), read_only=True)

    ok = True
    for qid, sql in queries.items():
        rel = con.sql(sql)
        rows = rel.fetchall()
        if len(rows) != 1:
            ok = False
            print(f"id {qid}: expected 1 row, got {len(rows)} -> {rows}")
            continue
        print(f"id {qid}: {dict(zip(rel.columns, rows[0]))}")

    with CSV_PATH.open(newline="") as f:
        figure_ids = {r["id"] for r in csv.DictReader(f) if r["type"] == "figure"}
    if figure_ids != set(queries):
        ok = False
        print(f"Missing SQL for ids: {sorted(figure_ids - set(queries), key=int)}")
        print(f"SQL without CSV row: {sorted(set(queries) - figure_ids, key=int)}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
