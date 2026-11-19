"""Conexión a la base de cifras financieras (DuckDB), siempre en solo lectura.

El agente solo consulta `finance.duckdb`; la base la escribe únicamente
`ingest/load_financials.py`. Abrirla con `read_only=True` hace que DuckDB
rechace cualquier INSERT, UPDATE, DELETE o CREATE.

Uso: `poetry run python -m financial_analyst_agent.sql.db` (consulta de prueba).
"""
from pathlib import Path

import duckdb

from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT

# =============================================================================
DB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'finance.duckdb'

SMOKE_SQL = 'SELECT company, count(*) AS n_rows FROM financials GROUP BY company ORDER BY company'


# =============================================================================
def connect(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Abre la base en solo lectura; falla si el archivo no existe.

    Sin la comprobación, DuckDB crearía una base vacía en una ruta errónea.
    """
    if not path.exists():
        raise FileNotFoundError(f'No existe {path}; genera los datos con `make data`')
    return duckdb.connect(str(path), read_only=True)


# =============================================================================
if __name__ == '__main__':
    with connect() as con:
        print(con.sql(SMOKE_SQL))
