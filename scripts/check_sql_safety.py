"""Casos de seguridad de la herramienta text-to-SQL, con una tabla de resultados.

Todas las consultas son inventadas: ninguna sale de `eval/questions.csv`, para
no inflar la medición de precisión del SQL. Las que pasan el validador se
ejecutan sobre `finance.duckdb` en solo lectura para comprobar que corren.

Uso: `poetry run python scripts/check_sql_safety.py` (sale con código 1 si
algún caso no da el resultado esperado).
"""
import sys
from dataclasses import dataclass

from financial_analyst_agent.sql import db
from financial_analyst_agent.sql.validator import SQLValidationError, validate_sql

# =============================================================================
PASS = 'pasa'


@dataclass
class Case:
    name: str
    sql: str
    expected: str  # PASS o el nombre de la excepción esperada


VALIDATOR_CASES = [
    Case('SELECT válido', "SELECT company, value FROM financials WHERE metric = 'Assets' AND fiscal_year = 2015", PASS),
    Case('SELECT con CTE', "WITH d AS (SELECT company, value FROM financials WHERE metric = 'Deposits') SELECT company, MAX(value) AS max_deposits FROM d GROUP BY company", PASS),
    Case('SELECT con ";" final', 'SELECT COUNT(*) AS n FROM financials;', PASS),
    Case("Cadena 'DROP' (sin falso positivo)", "SELECT company FROM financials WHERE metric = 'DROP'", PASS),
    Case('Columna last_update (sin falso positivo)', 'SELECT company, MAX(fiscal_year) AS last_update FROM financials GROUP BY company', PASS),
    Case('DROP TABLE', 'DROP TABLE financials', 'UnsafeSQLError'),
    Case('DELETE', 'DELETE FROM financials WHERE fiscal_year < 2010', 'UnsafeSQLError'),
    Case('UPDATE', "UPDATE financials SET value = 0 WHERE company = 'GS'", 'UnsafeSQLError'),
    Case('INSERT', "INSERT INTO financials (company) VALUES ('XYZ')", 'UnsafeSQLError'),
    Case('ATTACH', "ATTACH 'other.duckdb' AS other", 'UnsafeSQLError'),
    Case('COPY', "COPY financials TO 'dump.csv'", 'UnsafeSQLError'),
    Case('Dos sentencias', 'SELECT 1; DROP TABLE financials', 'UnsafeSQLError'),
    Case('UNION', "SELECT company FROM financials WHERE company = 'C' UNION SELECT company FROM financials", 'UnsafeSQLError'),
    Case('Texto no parseable', 'SELEC company FRM financials', 'InvalidSQLError'),
    Case('Salida vacía', '   ', 'InvalidSQLError'),
    Case('Solo ";"', ';', 'InvalidSQLError'),
]


# =============================================================================
def run_validator_case(case: Case) -> tuple[str, str]:
    """(resultado obtenido, detalle) de un caso del validador."""
    try:
        sql = validate_sql(case.sql)
    except SQLValidationError as exc:
        return type(exc).__name__, str(exc)

    with db.connect() as con:  # solo lectura
        n_rows = len(con.execute(sql).fetchmany(100))
    return PASS, f'{n_rows} filas; SQL: {" ".join(sql.split())}'


def print_table(rows: list[tuple[str, str, str, str, bool]]) -> None:
    print('| # | Caso | Esperado | Obtenido | OK | Detalle |')
    print('|---|------|----------|----------|----|---------|')
    for i, (name, expected, got, detail, ok) in enumerate(rows, 1):
        detail = detail.replace('|', '\\|')
        print(f"| {i} | {name} | {expected} | {got} | {'✅' if ok else '❌'} | {detail} |")


def main() -> int:
    rows = []
    for case in VALIDATOR_CASES:
        got, detail = run_validator_case(case)
        rows.append((case.name, case.expected, got, detail, got == case.expected))

    print_table(rows)
    failed = sum(not ok for *_, ok in rows)
    print(f'\n{len(rows) - failed}/{len(rows)} casos correctos')
    return 1 if failed else 0


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
