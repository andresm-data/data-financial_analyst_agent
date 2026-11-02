"""Carga las cifras financieras anuales de cada banco en DuckDB.

Usa la API XBRL de la SEC (company facts) como fuente de datos estructurados.
"""
import json
import logging
import time
from pathlib import Path

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
def main() -> None:
    """Descarga los company facts de cada banco de config/companies.yaml."""
    session = build_session()

    for bank in load_banks():
        download_company_facts(session, bank)


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
