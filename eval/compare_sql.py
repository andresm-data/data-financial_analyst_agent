"""Score a text-to-SQL run against the hand-written SQL in eval/figure_questions.sql.

Comparison rule (fixed before looking at sql_run1, approved 2026-11-24):

1. The expected values are the key columns (KEY_COLUMNS) of the result of the
   reference query for each id, run read-only (eval/figure_questions.sql, or
   eval/control_questions.sql for a run of the control questions). They are not parsed from the
   free text of questions.csv.
2. An answer is correct when all of these hold:
   - the tool returned no error;
   - it returned the same number of rows as the reference query;
   - if the expected answer names banks (ids 26-29, 31, 32, 37), the set of
     banks found in its cells equals that set (a cell is a bank if, stripped
     and case-insensitive, it is a ticker or a bank name from docs/schema.md).
     When no bank is expected (e.g. "What were Truist's deposits?"), a
     `company` column that repeats the filtered bank is not penalized
     (correction approved 2026-11-24, after sql_run2; sql_run1 unchanged);
   - every expected number matches at least one numeric cell, with a relative
     difference <= 1% (absolute difference <= 0.01 if the expected value is 0).
   Extra columns (e.g. intermediate values) are allowed.
3. No unit or scale normalization: 1013.2e9 does not match 1013.2 (USD bn),
   and 0.592 does not match 59.2 (percent).

Usage:
    poetry run python eval/compare_sql.py sql_run1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "finance.duckdb"
SQL_PATH = ROOT / "eval" / "figure_questions.sql"
SQL_FILES = {"figure": SQL_PATH, "control": ROOT / "eval" / "control_questions.sql"}
CSV_PATH = ROOT / "eval" / "questions.csv"
RESULTS_DIR = ROOT / "eval" / "results"

sys.path.insert(0, str(ROOT / "eval"))
from run_figure_queries import parse_queries  # noqa: E402

REL_TOL = 0.01
ABS_TOL_ZERO = 0.01

# Columns of the reference result that answer each question.
KEY_COLUMNS: dict[str, list[str]] = {
    "26": ["company"], "27": ["company"], "28": ["company"], "29": ["company"],
    "30": ["difference_usd_bn"],
    "31": ["higher", "difference_usd_bn"],
    "32": ["higher", "difference_usd_bn"],
    "33": ["deposits_to_assets_pct"],
    "34": ["growth_pct"],
    "35": ["change_usd_bn"],
    "36": ["change_pct"],
    "37": ["company"],
    "38": ["deposits_usd_bn"],
    "39": ["equity_usd_bn"],
    "40": ["net_income_usd_bn"],
    # Control questions (eval/control_questions.sql), fixed before running them.
    "41": ["liabilities_usd_bn"],
    "42": ["higher", "difference_usd_bn"],
    "43": ["liabilities_to_assets_pct"],
    "44": ["change_usd_bn"],
    "45": ["company"],
}

# Ticker or bank name (docs/schema.md) -> ticker.
BANKS = {
    "BAC": "BAC", "Bank of America": "BAC",
    "C": "C", "Citigroup": "C",
    "GS": "GS", "Goldman Sachs": "GS",
    "JPM": "JPM", "JPMorgan Chase": "JPM",
    "MS": "MS", "Morgan Stanley": "MS",
    "TFC": "TFC", "Truist Financial Corp": "TFC", "Truist Financial": "TFC",
}


def norm_text(value: str) -> str:
    """Case-insensitive, without spaces."""
    return "".join(str(value).split()).casefold()


BANK_LOOKUP = {norm_text(k): v for k, v in BANKS.items()}


def as_bank(value) -> str | None:
    return BANK_LOOKUP.get(norm_text(value)) if isinstance(value, str) else None


def as_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def numbers_match(expected: float, got: float) -> bool:
    if expected == 0:
        return abs(got) <= ABS_TOL_ZERO
    return abs(got - expected) / abs(expected) <= REL_TOL


def expected_answers(con: duckdb.DuckDBPyConnection, sql_path: Path = SQL_PATH) -> dict[str, dict]:
    """id -> {'n_rows', 'banks', 'numbers'} from the reference queries."""
    out = {}
    for qid, sql in parse_queries(sql_path.read_text()).items():
        rel = con.sql(sql)
        rows = [dict(zip(rel.columns, r)) for r in rel.fetchall()]
        banks, numbers = set(), []
        for row in rows:
            for col in KEY_COLUMNS[qid]:
                value = row[col]
                if as_bank(value):
                    banks.add(as_bank(value))
                else:
                    numbers.append(float(value))
        out[qid] = {"n_rows": len(rows), "banks": banks, "numbers": numbers}
    return out


def is_correct(record: dict, expected: dict) -> bool:
    """Apply the comparison rule to one run record."""
    if record["error"]:
        return False
    rows = record["rows"]
    if len(rows) != expected["n_rows"]:
        return False
    cells = [v for row in rows for v in row.values()]
    banks = {b for b in map(as_bank, cells) if b}
    if expected["banks"] and banks != expected["banks"]:
        return False
    got = [n for n in map(as_number, cells) if n is not None]
    return all(any(numbers_match(e, g) for g in got) for e in expected["numbers"])


def format_expected(expected: dict) -> str:
    parts = sorted(expected["banks"]) + [f"{n:g}" for n in expected["numbers"]]
    return "; ".join(parts)


def load_run(run_name: str) -> tuple[dict, list[dict]]:
    path = RESULTS_DIR / f"{run_name}_raw.jsonl"
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[0]["meta"], lines[1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_name", help="e.g. sql_run1, sql_run2")
    args = parser.parse_args()

    meta, records = load_run(args.run_name)
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        expected_text = {r["id"]: r["expected_answer"] for r in csv.DictReader(f)}
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        expected = expected_answers(con, SQL_FILES[meta.get("question_type", "figure")])

    out_path = RESULTS_DIR / f"{args.run_name}.csv"
    fields = ["id", "question", "sql", "answer", "expected_answer", "expected_values",
              "correct", "error", "intentos", "seconds", "run", "model", "temperature",
              "run_date", "commit"]
    n_correct = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            ok = is_correct(rec, expected[rec["id"]])
            n_correct += ok
            writer.writerow({
                "id": rec["id"], "question": rec["question"], "sql": rec["sql"],
                "answer": json.dumps(rec["rows"], default=str),
                "expected_answer": expected_text[rec["id"]],
                "expected_values": format_expected(expected[rec["id"]]),
                "correct": "yes" if ok else "no", "error": rec["error"] or "",
                "intentos": rec["attempts"], "seconds": rec["seconds"],
                "run": meta["run"], "model": meta["model"],
                "temperature": meta["temperature"], "run_date": meta["date"],
                "commit": meta["commit"],
            })
            print(f"id {rec['id']}: {'yes' if ok else 'no'}")

    pct = 100 * n_correct / len(records)
    print(f"{args.run_name}: {n_correct}/{len(records)} correct ({pct:.1f}%) -> "
          f"{out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
