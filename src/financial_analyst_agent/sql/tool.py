"""Herramienta de LangChain: pregunta -> SQL generado -> filas de DuckDB.

La única red de seguridad por ahora es la conexión de solo lectura
(`sql/db.py`): DuckDB rechaza cualquier escritura. Aún no hay validación del
SQL, LIMIT automático ni reintentos; si la consulta falla se devuelve el error.

Uso: `poetry run python -m financial_analyst_agent.sql.tool` (5 preguntas de
prueba, ninguna de `eval/questions.csv`).
"""
from functools import cache
from typing import Any

import duckdb
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

from financial_analyst_agent.sql import db
from financial_analyst_agent.sql.chain import build_sql_chain

# =============================================================================
MAX_ROWS = 100  # límite provisional hasta tener el LIMIT automático

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


def run_sql(sql: str, connect=db.connect) -> dict[str, Any]:
    """Ejecuta el SQL en solo lectura y devuelve hasta MAX_ROWS filas.

    Cada fila es un dict columna -> valor. Si DuckDB rechaza la consulta, se
    devuelve su mensaje en `error` y ninguna fila.
    """
    try:
        with connect() as con:
            cursor = con.execute(sql)
            columns = [col[0] for col in cursor.description or []]
            rows = cursor.fetchmany(MAX_ROWS) if columns else []
    except duckdb.Error as exc:
        return {'sql': sql, 'columns': [], 'rows': [],
                'error': f'{type(exc).__name__}: {exc}'}

    return {'sql': sql, 'columns': columns,
            'rows': [dict(zip(columns, row)) for row in rows], 'error': None}


def answer_with_sql(question: str, chain: Runnable | None = None, connect=db.connect) -> dict[str, Any]:
    """Genera el SQL de la pregunta con la cadena y lo ejecuta."""
    sql = (chain or sql_chain()).invoke({'question': question})
    return run_sql(sql, connect)


@tool
def text_to_sql(question: str) -> dict[str, Any]:
    """Answer a question about the annual figures (revenues, net income, assets,
    liabilities, stockholders' equity, deposits) of BAC, C, GS, JPM, MS and TFC
    from their 10-K filings, fiscal years 2006-2025.

    Writes a DuckDB SELECT query, runs it read-only and returns the SQL, the
    column names and up to 100 rows. All amounts are in US dollars. If the
    query fails, `error` holds the database message and `rows` is empty.
    """
    return answer_with_sql(question)


def main() -> None:
    """Pasa las 5 preguntas de prueba por la herramienta e imprime el resultado."""
    for i, question in enumerate(TEST_QUESTIONS, 1):
        result = text_to_sql.invoke({'question': question})
        print(f'\n=== {i}. {question}')
        print(f"SQL:\n{result['sql']}")
        print(f"Filas: {result['rows']}")
        print(f"Error: {result['error'] or 'no'}")


# =============================================================================
if __name__ == '__main__':
    main()
