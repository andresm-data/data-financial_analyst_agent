"""Herramienta de LangChain: pregunta -> SQL generado -> filas de DuckDB.

Barreras, de fuera hacia dentro:
1. `sql/validator.py` revisa el árbol de sqlglot: un único SELECT, sin
   comandos ni funciones prohibidos, y con LIMIT de 100 como máximo. Lo que se
   ejecuta y se devuelve es ese SQL validado.
2. `fetchmany(MAX_ROWS)`: nunca se leen más de 100 filas.
3. La conexión de `sql/db.py`: solo lectura y sin acceso a otros archivos.

Si la consulta falla al ejecutarse (o no se puede parsear), se envían al
modelo la pregunta, el SQL y el error para que lo corrija, y el nuevo SQL se
vuelve a validar: como máximo MAX_RETRIES reintentos (3 intentos en total).
Los rechazos de seguridad y los resultados vacíos (0 filas) no se reintentan.
Cada intento queda en el log `financial_analyst_agent.sql.tool` con su SQL,
su estado y el tipo de error, para poder agrupar los fallos por causa.

Uso: `poetry run python -m financial_analyst_agent.sql.tool` (5 preguntas de
prueba, ninguna de `eval/questions.csv`).
"""
import logging
from functools import cache
from typing import Any

import duckdb
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

from financial_analyst_agent.sql import db
from financial_analyst_agent.sql.chain import build_sql_chain, build_sql_repair_chain
from financial_analyst_agent.sql.validator import (
    MAX_LIMIT, InvalidSQLError, UnsafeSQLError, validate_sql,
)

# =============================================================================
logger = logging.getLogger(__name__)

MAX_ROWS = MAX_LIMIT  # segunda barrera detrás del LIMIT del validador
MAX_RETRIES = 2       # reintentos tras el primer intento: 3 intentos en total

TEST_QUESTIONS = [
    "What was Goldman Sachs' net income in fiscal year 2021?",
    'Which bank had the lowest total liabilities in fiscal year 2022?',
    'Which had more total deposits in fiscal year 2017, JPMorgan Chase or Citigroup, and by how much?',
    "What was the percentage change in Truist Financial's total assets from fiscal year 2019 to 2023?",
    "What was Bank of America's equity-to-assets ratio in fiscal year 2014?",
]


# =============================================================================
@cache
def sql_chain() -> Runnable:
    """La cadena se crea una sola vez, al usar la herramienta por primera vez."""
    return build_sql_chain()


@cache
def repair_chain() -> Runnable:
    """Cadena de corrección, creada solo si alguna consulta falla."""
    return build_sql_repair_chain()


def error_result(sql: str, error: str) -> dict[str, Any]:
    """Resultado sin filas con el mensaje de error."""
    return {'sql': sql, 'columns': [], 'rows': [], 'error': error}


def error_type(error: str) -> str:
    """Causa del error para agrupar en el log: `BinderException: ...` -> `BinderException`."""
    return error.split(':', 1)[0]


def log_attempt(question: str, attempt: int, sql: str, error: str | None, n_rows: int = 0) -> None:
    """Un registro por intento; los campos van también en `extra` para filtrarlos."""
    status = 'ok' if error is None else ('rejected' if error.startswith('UnsafeSQLError') else 'failed')
    fields = {'question': question, 'attempt': attempt, 'status': status, 'sql': sql,
              'error_type': error_type(error) if error else None, 'error': error, 'n_rows': n_rows}
    logger.log(
        logging.INFO if error is None else logging.WARNING,
        'text_to_sql attempt=%d status=%s error_type=%s n_rows=%d sql=%r error=%r',
        attempt, status, fields['error_type'], n_rows, sql, error, extra={'sql_attempt': fields},
    )


def run_sql(sql: str, connect=db.connect) -> dict[str, Any]:
    """Ejecuta el SQL en solo lectura y devuelve hasta MAX_ROWS filas.

    No valida: recibe SQL que ya pasó por `validate_sql`. Cada fila es un dict
    columna -> valor. Si DuckDB rechaza la consulta, se devuelve su mensaje en
    `error` y ninguna fila.
    """
    try:
        with connect() as con:
            cursor = con.execute(sql)
            columns = [col[0] for col in cursor.description or []]
            rows = cursor.fetchmany(MAX_ROWS) if columns else []
    except duckdb.Error as exc:
        return error_result(sql, f'{type(exc).__name__}: {exc}')

    return {'sql': sql, 'columns': columns,
            'rows': [dict(zip(columns, row)) for row in rows], 'error': None}


def answer_with_sql(
    question: str,
    chain: Runnable | None = None,
    connect=db.connect,
    repair: Runnable | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Genera el SQL, lo valida, lo ejecuta y, si falla, pide una corrección.

    - Rechazo de seguridad (`UnsafeSQLError`): se devuelve de inmediato.
    - SQL no parseable o error de DuckDB: se pide al modelo una versión
      corregida, hasta `max_retries` veces.
    - 0 filas no es un error: se devuelve tal cual.

    El resultado incluye `attempts` (número de intentos usados).
    """
    sql = (chain or sql_chain()).invoke({'question': question})

    for attempt in range(1, max_retries + 2):
        try:
            safe_sql = validate_sql(sql)
        except UnsafeSQLError as exc:
            result = error_result(sql, f'UnsafeSQLError: {exc}')
            log_attempt(question, attempt, sql, result['error'])
            return {**result, 'attempts': attempt}
        except InvalidSQLError as exc:
            result = error_result(sql, f'InvalidSQLError: {exc}')
        else:
            result = run_sql(safe_sql, connect)

        log_attempt(question, attempt, result['sql'], result['error'], len(result['rows']))
        if result['error'] is None:
            return {**result, 'attempts': attempt}
        if attempt > max_retries:
            break
        sql = (repair or repair_chain()).invoke(
            {'question': question, 'sql': result['sql'], 'error': result['error']},
        )

    return {**result, 'attempts': attempt,
            'error': f'La consulta falló tras {attempt} intentos. Último error: {result["error"]}'}


@tool
def text_to_sql(question: str) -> dict[str, Any]:
    """Answer a question about the annual figures (revenues, net income, assets,
    liabilities, stockholders' equity, deposits) of BAC, C, GS, JPM, MS and TFC
    from their 10-K filings, fiscal years 2006-2025.

    Writes a DuckDB SELECT query, checks it is a single safe SELECT, caps it
    at LIMIT 100, runs it read-only and returns the executed SQL, the column
    names and the rows. Amounts are in USD billions and ratios or changes in
    percent. A query that fails is corrected and retried up to 2 times. If it is rejected as unsafe or still
    fails, `error` holds the reason and `rows` is empty; never guess figures.
    """
    return answer_with_sql(question)


def main() -> None:
    """Pasa las 5 preguntas de prueba por la herramienta e imprime el resultado."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    for i, question in enumerate(TEST_QUESTIONS, 1):
        result = text_to_sql.invoke({'question': question})
        print(f'\n=== {i}. {question}')
        print(f"SQL:\n{result['sql']}")
        print(f"Filas: {result['rows']}")
        print(f"Error: {result['error'] or 'no'}")


# =============================================================================
if __name__ == '__main__':
    main()
