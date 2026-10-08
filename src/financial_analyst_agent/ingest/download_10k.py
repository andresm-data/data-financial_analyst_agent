"""Descarga los 3 informes 10-K más recientes de cada banco desde SEC EDGAR."""
import os
import sys
import time
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
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_10K_DIR = PROJECT_ROOT / 'data' / 'raw' / '10k'

SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'
ARCHIVE_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}'

FORM_TYPE = '10-K'
FILINGS_PER_BANK = 3
REQUEST_DELAY = 0.3


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


# =============================================================================
def download_filings(session: requests.Session, filings: pd.DataFrame) -> None:
    """Descarga los documentos de los informes y los guarda en data/raw/10k/.

    Cada archivo se guarda como `{ticker}_{filingDate}_{primaryDocument}`. Si
    ya existe se omite, y entre descargas se esperan 0,3 segundos para
    respetar el límite de peticiones de la SEC.

    Args:
        session (requests.Session): Sesión con el User-Agent que exige la SEC.
        filings (pd.DataFrame): Informes a descargar con las columnas
            `ticker`, `filingDate`, `primaryDocument` y `url`.
    """
    RAW_10K_DIR.mkdir(parents=True, exist_ok=True)

    for r in filings.itertuples(index=False):
        # Prefijo con ticker y fecha para evitar colisiones
        target = RAW_10K_DIR / f'{r.ticker}_{r.filingDate}_{r.primaryDocument}'

        if target.exists():
            print(f'  ya existe {target.name}, se omite')
            continue

        response = session.get(r.url, timeout=60)
        response.raise_for_status()

        target.write_bytes(response.content)
        print(f'  descargado {target.name}')

        time.sleep(REQUEST_DELAY)


# =============================================================================
def main() -> None:
    """Ejecuta el flujo completo de descarga de los 10-K de cada banco.

    1. Consulta la lista de informes de cada banco de config/companies.yaml.
    2. Se queda con los 3 informes 10-K más recientes por banco, les agrega
       la URL de descarga y guarda la tabla en data/processed/filings.csv.
    3. Descarga cada documento a data/raw/10k/.
    """
    session = build_session()
    banks = load_banks()

    selected = []

    for bank in banks:
        print(f'Consultando {bank["ticker"]} (CIK {bank["cik"]})')

        filings = fetch_recent_filings(session, bank['cik'])
        selected.append(latest_10k(filings, bank))

        time.sleep(REQUEST_DELAY)

    table = pd.concat(selected, ignore_index=True)
    table['url'] = [
        build_document_url(r.cik, r.accessionNumber, r.primaryDocument)
        for r in table.itertuples(index=False)
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / 'filings.csv'

    table.to_csv(csv_path, index=False)

    print(f'Tabla guardada en {csv_path} ({len(table)} filas)')
    print("Descargando documentos")

    download_filings(session, table)


# =============================================================================
if __name__ == "__main__":
    main()
