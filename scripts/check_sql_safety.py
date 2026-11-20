"""Casos de seguridad de la herramienta text-to-SQL, con una tabla de resultados.

Todas las consultas son inventadas: ninguna sale de `eval/questions.csv`, para
no inflar la medición de precisión del SQL. Las que pasan el validador se
ejecutan sobre `finance.duckdb` en solo lectura para comprobar que corren y
que el SQL final tiene LIMIT de 100 como máximo.

Uso: `poetry run python scripts/check_sql_safety.py` (sale con código 1 si
algún caso no da el resultado esperado).
"""
import logging
import sys
from dataclasses import dataclass

import duckdb
import sqlglot
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_analyst_agent.sql import db, tool
from financial_analyst_agent.sql.chain import build_sql_chain, build_sql_repair_chain
from financial_analyst_agent.sql.validator import (
    DIALECT, MAX_LIMIT, SQLValidationError, limit_value, validate_sql,
)

# =============================================================================
PASS = 'pasa'


@dataclass
class Case:
    name: str
    sql: str
    expected: str           # PASS o el nombre de la excepción esperada
    message: str = ''       # texto que debe aparecer en el mensaje de rechazo
    limit: int | None = None  # LIMIT esperado en el SQL final (si pasa)


VALIDATOR_CASES = [
    # --- Consultas legítimas
    Case('SELECT válido', "SELECT company, value FROM financials WHERE metric = 'Assets' AND fiscal_year = 2015", PASS, limit=100),
    Case('SELECT con CTE', "WITH d AS (SELECT company, value FROM financials WHERE metric = 'Deposits') SELECT company, MAX(value) AS max_deposits FROM d GROUP BY company", PASS, limit=100),
    Case('SELECT con ";" final', 'SELECT COUNT(*) AS n FROM financials;', PASS, limit=100),
    Case("Cadena 'DROP' (sin falso positivo)", "SELECT company FROM financials WHERE metric = 'DROP'", PASS, limit=100),
    Case('Columna last_update (sin falso positivo)', 'SELECT company, MAX(fiscal_year) AS last_update FROM financials GROUP BY company', PASS, limit=100),
    Case("Cadena 'read_csv' (sin falso positivo)", "SELECT company FROM financials WHERE metric = 'read_csv(x)'", PASS, limit=100),
    # --- LIMIT
    Case('Sin LIMIT -> LIMIT 100', 'SELECT company, fiscal_year, value FROM financials', PASS, limit=100),
    Case('LIMIT 5 -> igual', 'SELECT company, value FROM financials ORDER BY value DESC LIMIT 5', PASS, limit=5),
    Case('LIMIT 5000 -> 100', 'SELECT company, value FROM financials LIMIT 5000', PASS, limit=100),
    Case('LIMIT 100 -> igual', 'SELECT company, value FROM financials LIMIT 100', PASS, limit=100),
    Case('LIMIT 5000 OFFSET 10 -> 100, conserva OFFSET', 'SELECT company, value FROM financials LIMIT 5000 OFFSET 10', PASS, limit=100),
    Case('LIMIT 50% -> 100', 'SELECT company, value FROM financials LIMIT 50%', PASS, limit=100),
    Case('LIMIT en subconsulta no cuenta', 'SELECT * FROM (SELECT company, value FROM financials LIMIT 3) AS t', PASS, limit=100),
    # --- Comandos prohibidos
    Case('DROP TABLE', 'DROP TABLE financials', 'UnsafeSQLError', 'DROP no está permitido'),
    Case('DELETE', 'DELETE FROM financials WHERE fiscal_year < 2010', 'UnsafeSQLError', 'DELETE no está permitido'),
    Case('UPDATE', "UPDATE financials SET value = 0 WHERE company = 'GS'", 'UnsafeSQLError', 'UPDATE no está permitido'),
    Case('INSERT', "INSERT INTO financials (company) VALUES ('XYZ')", 'UnsafeSQLError', 'INSERT no está permitido'),
    Case('ATTACH', "ATTACH 'other.duckdb' AS other", 'UnsafeSQLError', 'ATTACH no está permitido'),
    Case('COPY', "COPY financials TO 'dump.csv'", 'UnsafeSQLError', 'COPY no está permitido'),
    Case('DELETE dentro de un CTE', 'WITH gone AS (DELETE FROM financials RETURNING *) SELECT * FROM gone', 'UnsafeSQLError', 'dentro de un CTE'),
    Case('Dos sentencias', 'SELECT 1; DROP TABLE financials', 'UnsafeSQLError', 'una sola sentencia'),
    Case('UNION', "SELECT company FROM financials WHERE company = 'C' UNION SELECT company FROM financials", 'UnsafeSQLError', 'UNION'),
    Case('Otra sentencia (PRAGMA)', 'PRAGMA database_list', 'UnsafeSQLError', 'Solo se permiten consultas SELECT'),
    # --- Lectura de archivos desde un SELECT
    Case('SELECT con read_csv', "SELECT * FROM read_csv('/etc/passwd')", 'UnsafeSQLError', 'read_csv()'),
    Case('read_parquet en subconsulta', "SELECT company FROM financials WHERE company IN (SELECT c FROM read_parquet('x.parquet'))", 'UnsafeSQLError', 'read_parquet()'),
    Case('read_text en CTE', "WITH f AS (SELECT * FROM read_text('~/.ssh/id_rsa')) SELECT * FROM f", 'UnsafeSQLError', 'read_text()'),
    Case('glob', "SELECT * FROM glob('/home/*')", 'UnsafeSQLError', 'glob()'),
    Case("FROM 'archivo.csv'", "SELECT * FROM 'data/processed/filings.csv'", 'UnsafeSQLError', 'ruta de archivo'),
    # --- Texto no válido
    Case('Texto no parseable', 'SELEC company FRM financials', 'InvalidSQLError', 'No se pudo parsear'),
    Case('Salida vacía', '   ', 'InvalidSQLError', 'vacía'),
    Case('Solo ";"', ';', 'InvalidSQLError', 'vacía'),
]

# Consultas que se ejecutan SIN validador, para probar la barrera de la conexión.
CONNECTION_CASES = [
    ('Conexión: SELECT normal', 'SELECT COUNT(*) FROM financials', PASS),
    ('Conexión: read_csv sin validador', "SELECT * FROM read_csv('data/processed/filings.csv')", 'PermissionException'),
    ('Conexión: COPY sin validador', "COPY financials TO 'leak.csv'", 'PermissionException'),
    ('Conexión: CREATE sin validador', 'CREATE TABLE x (a INTEGER)', 'InvalidInputException'),
]

# Reintentos con un modelo falso: respuestas en orden (generación y luego
# correcciones). Esperado: (¿éxito?, intentos, filas o None si no importa).
GOOD_SQL = 'SELECT company, COUNT(*) AS n FROM financials GROUP BY company ORDER BY company'
RETRY_CASES = [
    ('Reintento: 1.º mal (columna), 2.º bien', ['SELECT nope FROM financials', GOOD_SQL], (True, 2, 6)),
    ('Reintento: 3 SQL mal -> para tras 2 reintentos',
     ['SELECT nope FROM financials', 'SELEC company FRM financials', 'SELECT company FROM missing_table'],
     (False, 3, 0)),
    ('Reintento: rechazo de seguridad no se reintenta', ['DROP TABLE financials', GOOD_SQL], (False, 1, 0)),
    ('Reintento: corrección insegura se rechaza ya', ['SELECT nope FROM financials', "SELECT * FROM read_csv('x.csv')", GOOD_SQL], (False, 2, 0)),
    ('Reintento: 0 filas no es error', ["SELECT company FROM financials WHERE company = 'XYZ'", GOOD_SQL], (True, 1, 0)),
]
UNUSED = "SELECT 'respuesta de relleno: el modelo no debería llegar aquí'"


class ListHandler(logging.Handler):
    """Guarda los registros de los intentos para comprobarlos."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


# =============================================================================
def one_line(sql: str) -> str:
    return ' '.join(sql.split())


def run_validator_case(case: Case) -> tuple[str, str, bool]:
    """(resultado obtenido, detalle, ok) de un caso del validador."""
    try:
        sql = validate_sql(case.sql)
    except SQLValidationError as exc:
        got = type(exc).__name__
        return got, str(exc), got == case.expected and case.message in str(exc)

    with db.connect() as con:  # solo lectura
        n_rows = len(con.execute(sql).fetchmany(MAX_LIMIT))
    limit = limit_value(sqlglot.parse_one(sql, dialect=DIALECT).args.get('limit'))
    ok = case.expected == PASS and limit == case.limit and limit <= MAX_LIMIT
    return PASS, f'LIMIT {limit}; {n_rows} filas; SQL: {one_line(sql)}', ok


def run_connection_case(sql: str, expected: str) -> tuple[str, str, bool]:
    """Ejecuta directamente sobre la conexión, sin validador."""
    try:
        with db.connect() as con:
            con.execute(sql).fetchall()
    except duckdb.Error as exc:
        got = type(exc).__name__
        return got, str(exc).splitlines()[0], got == expected
    return PASS, 'ejecutada', expected == PASS


def run_retry_case(responses: list[str], expected: tuple[bool, int, int]) -> tuple[str, str, bool]:
    """Pasa una pregunta inventada por la herramienta con un modelo falso."""
    # La respuesta de relleno evita que el modelo falso vuelva a empezar la
    # lista, así `llm.i` es exactamente el número de llamadas al modelo.
    llm = FakeListChatModel(responses=[*responses, UNUSED])
    handler = ListHandler()
    tool.logger.addHandler(handler)
    tool.logger.setLevel(logging.INFO)  # los intentos correctos se registran en INFO
    try:
        result = tool.answer_with_sql(
            'How many rows does each bank have?',  # inventada, no está en eval/questions.csv
            chain=build_sql_chain(llm), repair=build_sql_repair_chain(llm),
        )
    finally:
        tool.logger.removeHandler(handler)

    ok_expected, attempts_expected, rows_expected = expected
    success = result['error'] is None
    logged = [r.sql_attempt for r in handler.records]
    ok = (success == ok_expected and result['attempts'] == attempts_expected == llm.i == len(logged)
          and len(result['rows']) == rows_expected)
    got = f"{'éxito' if success else 'error'}, {result['attempts']} intentos"
    causes = ', '.join(f"{a['attempt']}:{a['status']}/{a['error_type'] or '-'}" for a in logged)
    detail = f"llamadas al modelo: {llm.i}; filas: {len(result['rows'])}; log: [{causes}]"
    if not success:
        detail += f"; error: {result['error'].splitlines()[0]}"
    return got, detail, ok


def print_table(rows: list[tuple[str, str, str, str, bool]]) -> None:
    print('| # | Caso | Esperado | Obtenido | OK | Detalle |')
    print('|---|------|----------|----------|----|---------|')
    for i, (name, expected, got, detail, ok) in enumerate(rows, 1):
        detail = detail.replace('|', '\\|')
        print(f"| {i} | {name} | {expected} | {got} | {'✅' if ok else '❌'} | {detail} |")


def main() -> int:
    rows = []
    for case in VALIDATOR_CASES:
        expected = case.expected if case.expected != PASS else f'pasa (LIMIT {case.limit})'
        rows.append((case.name, expected, *run_validator_case(case)))
    for name, sql, expected in CONNECTION_CASES:
        rows.append((name, expected, *run_connection_case(sql, expected)))
    for name, responses, expected in RETRY_CASES:
        label = f"{'éxito' if expected[0] else 'error'}, {expected[1]} intentos"
        rows.append((name, label, *run_retry_case(responses, expected)))

    print_table(rows)
    failed = sum(not ok for *_, ok in rows)
    print(f'\n{len(rows) - failed}/{len(rows)} casos correctos')
    return 1 if failed else 0


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
