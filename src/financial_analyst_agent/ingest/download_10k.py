"""Descarga los 3 informes 10-K más recientes de cada banco desde SEC EDGAR."""
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


# =============================================================================
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]

COMPANIES_FILE = PACKAGE_DIR / 'config' / 'companies.yaml'
ENV_FILE = PACKAGE_DIR / '.env'

SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'
ARCHIVE_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}'

FORM_TYPE = '10-K'
FILINGS_PER_BANK = 3


# =============================================================================
def load_banks() -> list[dict]:
    """Carga la lista de bancos a analizar desde config/companies.yaml.

    Returns:
        list[dict]: Un diccionario por banco con las claves `ticker`,
            `name`, `info` y `cik` (CIK como texto de 10 dígitos).
    """
    with COMPANIES_FILE.open(encoding='utf-8') as f:
        return yaml.safe_load(f)['banks']


# =============================================================================
def to_ascii(text: str) -> str:
    """La SEC responde 403 si el User-Agent tiene caracteres no ASCII."""
    return (
        unicodedata.normalize('NFKD', text)
                   .encode('ascii', 'ignore')
                   .decode('ascii')
    )


# =============================================================================
def build_session() -> requests.Session:
    """Crea una sesión HTTP con los encabezados que exige la SEC.

    Returns:
        requests.Session: Sesión con los encabezados `User-Agent` y
            `Accept-Encoding` para consultar EDGAR.

    Raises:
        SystemExit: Si `SEC_USER_AGENT` no está definido o está vacío.
    """
    # Cargar archivo .env
    load_dotenv(ENV_FILE)
    user_agent = os.getenv('SEC_USER_AGENT')

    if not user_agent:
        sys.exit(f'Falta SEC_USER_AGENT en {ENV_FILE}')

    session = requests.Session()
    session.headers.update({
        'User-Agent': to_ascii(user_agent),
        'Accept-Encoding': 'gzip, deflate'
    })

    return session


# =============================================================================
def fetch_recent_filings(session: requests.Session, cik: str) -> pd.DataFrame:
    """Obtiene la lista de informes recientes de un banco desde la API de la SEC.

    Args:
        session (requests.Session): Sesión con el User-Agent que exige la SEC.
        cik (str): CIK del banco; si tiene menos de 10 dígitos se completa
            con ceros a la izquierda.

    Returns:
        pd.DataFrame: Una fila por informe con las columnas `form`,
            `accessionNumber`, `primaryDocument` y `filingDate`, armadas
            a partir de las listas paralelas de `filings.recent`.
    """
    url = SUBMISSIONS_URL.format(cik=cik.zfill(10))
    response = session.get(url, timeout=30)
    response.raise_for_status()

    recent = response.json()['filings']['recent']
    columns = ['form', 'accessionNumber', 'primaryDocument', 'filingDate']

    return pd.DataFrame({
        col: recent[col] for col in columns
    })


# =============================================================================
def latest_10k(filings: pd.DataFrame, bank: dict) -> pd.DataFrame:
    """Selecciona los 3 informes 10-K más recientes de un banco.

    Args:
        filings (pd.DataFrame): Informes del banco tal como los devuelve
            `fetch_recent_filings`.
        bank (dict): Datos del banco con las claves `ticker`, `name` y `cik`.

    Returns:
        pd.DataFrame: Hasta 3 filas con `form == "10-K"`, ordenadas de la más
            reciente a la más antigua por `filingDate`, con las columnas
            `ticker`, `name` y `cik` agregadas al inicio.
    """
    selected = (
        filings[filings['form'] == FORM_TYPE]
        .sort_values('filingDate', ascending=False)
        .head(FILINGS_PER_BANK)
        .copy()
    )

    selected.insert(0, 'ticker', bank['ticker'])
    selected.insert(1, 'name', bank['name'])
    selected.insert(2, 'cik', bank['cik'])

    return selected


# =============================================================================
def build_document_url(
    cik: str, accession_number: str, primary_document: str
) -> str:
    """Construye la URL de descarga de un informe en el archivo EDGAR de la SEC.

    Args:
        cik (str): CIK del banco; se le quitan los ceros a la izquierda.
        accession_number (str): Número de registro del informe con guiones.
        primary_document (str): Nombre del documento principal del informe.

    Returns:
        str: URL con la forma
            `https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}`.
    """
    return ARCHIVE_URL.format(
        cik=int(cik),
        accession=accession_number.replace('-', ''),
        document=primary_document
    )
