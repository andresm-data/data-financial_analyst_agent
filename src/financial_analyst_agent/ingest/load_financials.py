"""Carga las cifras financieras anuales de cada banco en DuckDB.

Usa la API XBRL de la SEC (company facts) como fuente de datos estructurados.
"""
import json
import logging
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

from financial_analyst_agent.ingest.download_10k import (
    PROJECT_ROOT,
    build_session,
    load_banks,
)

logger = logging.getLogger(__name__)


# =============================================================================
RAW_FACTS_DIR = PROJECT_ROOT / 'data' / 'raw' / 'facts'

COMPANY_FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'

# La SEC permite hasta 10 peticiones por segundo; 0,15 s deja margen
REQUEST_DELAY = 0.15

FORM_TYPE = '10-K'
FISCAL_PERIOD = 'FY'
UNIT = 'USD'

# Etiquetas us-gaap por métrica, en orden de preferencia. Para cada año
# fiscal se usa la primera que tenga dato.
METRIC_TAGS = {
    'Revenues': [
        'Revenues',
        'RevenuesNetOfInterestExpense',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'InterestAndDividendIncomeOperating',
    ],
    'NetIncomeLoss': ['NetIncomeLoss', 'ProfitLoss'],
    'Assets': ['Assets'],
    'Liabilities': ['Liabilities'],
    'StockholdersEquity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'Deposits': ['Deposits'],
}

# Rango de días que se acepta como periodo anual en métricas de flujo
# (ingresos, utilidad). Excluye trimestres que vienen dentro del 10-K.
ANNUAL_DAYS = (330, 400)

COLUMNS = [
    'company', 'metric', 'fiscal_year', 'value', 'unit', 'period_end',
    'filed_date',
]

DB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'finance.duckdb'

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS financials (
    company     VARCHAR NOT NULL,
    metric      VARCHAR NOT NULL,
    fiscal_year INTEGER NOT NULL,
    value       DOUBLE  NOT NULL,
    unit        VARCHAR NOT NULL,
    period_end  DATE    NOT NULL,
    filed_date  DATE    NOT NULL,
    PRIMARY KEY (company, metric, fiscal_year)
)
"""

VERIFY_SQL = (
    'SELECT company, count(*) FROM financials GROUP BY company ORDER BY company'
)


# =============================================================================
def facts_path(ticker: str) -> Path:
    """Ruta del JSON de company facts de un banco en data/raw/facts/."""
    return RAW_FACTS_DIR / f'{ticker}.json'


# =============================================================================
def download_company_facts(
    session: requests.Session, bank: dict
) -> Path:
    """Descarga el JSON de company facts de un banco si aún no existe.

    Args:
        session (requests.Session): Sesión con el User-Agent que exige la SEC.
        bank (dict): Datos del banco con las claves `ticker` y `cik`.

    Returns:
        Path: Ruta del JSON guardado (o del que ya existía).
    """
    target = facts_path(bank['ticker'])

    if target.exists():
        logger.info('Ya existe %s, se omite', target.name)
        return target

    url = COMPANY_FACTS_URL.format(cik=str(bank['cik']).zfill(10))
    response = session.get(url, timeout=60)
    response.raise_for_status()

    RAW_FACTS_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    logger.info('Descargado %s', target.name)

    time.sleep(REQUEST_DELAY)

    return target


# =============================================================================
def load_company_facts(path: Path) -> dict:
    """Lee un JSON de company facts guardado en disco."""
    with path.open(encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
def annual_facts(facts: dict, tag: str) -> pd.DataFrame:
    """Devuelve los valores anuales de una etiqueta us-gaap, uno por año.

    Filtra `form == "10-K"` y `fp == "FY"`. El año fiscal sale de la fecha
    `end` del dato y no de `fy`, porque `fy` es el año del informe y un 10-K
    trae también cifras de años anteriores. Además:

    - En métricas de flujo (con `start`) solo acepta periodos de ~12 meses.
    - En saldos (sin `start`) solo acepta fechas que coinciden con el cierre
      fiscal más común de la etiqueta (p. ej. 12-31), para descartar saldos
      de apertura u otras fechas intermedias.
    - Si hay varias filas para el mismo año (reexpresiones), conserva la de
      `filed` más reciente.

    Args:
        facts (dict): JSON completo de company facts.
        tag (str): Etiqueta dentro de `facts > us-gaap`.

    Returns:
        pd.DataFrame: Columnas `fiscal_year`, `value`, `unit`, `period_end`
            y `filed_date`; vacío si la etiqueta no existe o no tiene datos.
    """
    empty = pd.DataFrame(
        columns=['fiscal_year', 'value', 'unit', 'period_end', 'filed_date']
    )
    rows = (
        facts.get('facts', {}).get('us-gaap', {})
             .get(tag, {}).get('units', {}).get(UNIT, [])
    )
    df = pd.DataFrame(rows)

    if df.empty:
        return empty

    df = df[(df['form'] == FORM_TYPE) & (df['fp'] == FISCAL_PERIOD)].copy()

    if df.empty:
        return empty

    df['end'] = pd.to_datetime(df['end'])

    if 'start' in df.columns and df['start'].notna().any():
        days = (df['end'] - pd.to_datetime(df['start'])).dt.days
        df = df[days.between(*ANNUAL_DAYS)]
    else:
        month_day = df['end'].dt.strftime('%m-%d')
        df = df[month_day == month_day.mode().iloc[0]]

    if df.empty:
        return empty

    df['fiscal_year'] = df['end'].dt.year
    df = (
        df.sort_values(['filed', 'accn'])
          .drop_duplicates('fiscal_year', keep='last')
          .sort_values('fiscal_year')
    )

    return pd.DataFrame({
        'fiscal_year': df['fiscal_year'].astype(int),
        'value': df['val'].astype(float),
        'unit': UNIT,
        'period_end': df['end'].dt.date,
        'filed_date': pd.to_datetime(df['filed']).dt.date,
    })


# =============================================================================
def extract_metric(
    facts: dict, metric: str, tags: list[str]
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Extrae una métrica probando sus etiquetas en orden de preferencia.

    Para cada año fiscal usa la primera etiqueta de `tags` que tenga dato,
    así un año sin la etiqueta principal se completa con una alternativa.

    Args:
        facts (dict): JSON completo de company facts.
        metric (str): Nombre de la métrica en la tabla final.
        tags (list[str]): Etiquetas us-gaap en orden de preferencia.

    Returns:
        tuple[pd.DataFrame, dict[str, list[int]]]: Las filas anuales de la
            métrica y, por cada etiqueta usada, los años que aportó.
    """
    frames = []
    used: dict[str, list[int]] = {}
    covered: set[int] = set()

    for tag in tags:
        df = annual_facts(facts, tag)
        df = df[~df['fiscal_year'].isin(covered)]

        if df.empty:
            continue

        years = df['fiscal_year'].tolist()
        used[tag] = years
        covered.update(years)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=COLUMNS), used

    result = pd.concat(frames, ignore_index=True).sort_values('fiscal_year')
    result.insert(0, 'metric', metric)

    return result, used


# =============================================================================
def format_years(years: list[int]) -> str:
    """Resume una lista de años como `2009-2025 (17 años)`."""
    return f'{min(years)}-{max(years)} ({len(years)} años)'


# =============================================================================
def extract_financials(facts: dict, company: str) -> pd.DataFrame:
    """Extrae las 6 métricas anuales de un banco.

    Muestra por consola la etiqueta de ingresos usada, una nota por cada
    métrica que use una etiqueta distinta de la principal y una advertencia por cada métrica sin datos.

    Args:
        facts (dict): JSON completo de company facts del banco.
        company (str): Ticker del banco; va en la columna `company`.

    Returns:
        pd.DataFrame: Filas con las columnas de `COLUMNS`.
    """
    frames = []

    for metric, tags in METRIC_TAGS.items():
        df, used = extract_metric(facts, metric, tags)

        if df.empty:
            logger.warning('%s: sin datos para %s', company, metric)
            continue

        # Los ingresos siempre se reportan; el resto solo si hay alternativa
        if metric == 'Revenues' or list(used) != [tags[0]]:
            detail = ', '.join(
                f'{tag} {format_years(years)}' for tag, years in used.items()
            )
            logger.info('%s: %s usa %s', company, metric, detail)

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result.insert(0, 'company', company)

    return result[COLUMNS]


# =============================================================================
def load_to_duckdb(table: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Carga las cifras en la tabla `financials` de DuckDB.

    La carga es repetible: en una sola transacción borra las filas de las
    empresas que vienen en `table` y vuelve a insertarlas. La clave primaria
    (company, metric, fiscal_year) impide duplicados.

    Args:
        table (pd.DataFrame): Filas con las columnas de `COLUMNS`.
        db_path (Path): Ruta del archivo .duckdb.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as con:
        con.execute(CREATE_TABLE_SQL)
        con.register('new_rows', table[COLUMNS])

        con.execute('BEGIN TRANSACTION')
        try:
            con.execute(
                'DELETE FROM financials '
                'WHERE company IN (SELECT DISTINCT company FROM new_rows)'
            )
            con.execute(
                f'INSERT INTO financials ({", ".join(COLUMNS)}) '
                f'SELECT {", ".join(COLUMNS)} FROM new_rows'
            )
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK')
            raise

    logger.info('Cargadas %d filas en %s', len(table), db_path)


# =============================================================================
def verify(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Cuenta las filas por empresa en la tabla `financials`."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(VERIFY_SQL).df()


# =============================================================================
def main() -> None:
    """Ejecuta el flujo completo.

    1. Descarga los company facts de cada banco de config/companies.yaml.
    2. Extrae las 6 métricas anuales.
    3. Las carga en data/processed/finance.duckdb y muestra el conteo.
    """
    session = build_session()
    frames = []

    for bank in load_banks():
        path = download_company_facts(session, bank)
        facts = load_company_facts(path)
        frames.append(extract_financials(facts, bank['ticker']))

    table = pd.concat(frames, ignore_index=True)
    load_to_duckdb(table)

    print(verify().to_string(index=False))


# =============================================================================
def run() -> None:
    """Configura los logs y ejecuta main; es el punto de entrada del comando."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    main()


# =============================================================================
if __name__ == '__main__':
    run()
