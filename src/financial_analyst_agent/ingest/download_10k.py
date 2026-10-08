"""Descarga los 3 informes 10-K más recientes de cada banco desde SEC EDGAR."""
import logging
import os
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# =============================================================================
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]

COMPANIES_FILE = PACKAGE_DIR / 'config' / 'companies.yaml'
ENV_FILE = PACKAGE_DIR / '.env'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_10K_DIR = PROJECT_ROOT / 'data' / 'raw' / '10k'

SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'
SUBMISSIONS_PAGE_URL = 'https://data.sec.gov/submissions/{name}'
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
def filings_to_frame(data: dict) -> pd.DataFrame:
    """Convierte las listas paralelas de informes de la SEC en un DataFrame.

    Args:
        data (dict): Diccionario con las listas paralelas `form`,
            `accessionNumber`, `primaryDocument`, `filingDate` y
            `reportDate`.

    Returns:
        pd.DataFrame: Una fila por informe con esas cinco columnas.
    """
    columns = [
        'form', 'accessionNumber', 'primaryDocument', 'filingDate', 'reportDate'
    ]

    return pd.DataFrame({
        col: data[col] for col in columns
    })


# =============================================================================
def fetch_recent_filings(session: requests.Session, cik: str) -> pd.DataFrame:
    """Obtiene la lista de informes de un banco desde la API de la SEC.

    En bancos con muchas presentaciones, `filings.recent` solo cubre el
    último año. Si ahí hay menos de 3 informes 10-K, recorre las páginas de
    `filings.files` (de la más nueva a la más vieja) hasta completarlos o
    hasta que no queden páginas.

    Args:
        session (requests.Session): Sesión con el User-Agent que exige la SEC.
        cik (str): CIK del banco; si tiene menos de 10 dígitos se completa
            con ceros a la izquierda.

    Returns:
        pd.DataFrame: Una fila por informe con las columnas `form`,
            `accessionNumber`, `primaryDocument`, `filingDate` y
            `reportDate`, armadas
            a partir de las listas paralelas de `filings.recent` y de las
            páginas adicionales consultadas.
    """
    url = SUBMISSIONS_URL.format(cik=cik.zfill(10))
    response = session.get(url, timeout=30)
    response.raise_for_status()

    filings = response.json()['filings']
    frames = [filings_to_frame(filings['recent'])]
    found = (frames[0]['form'] == FORM_TYPE).sum()

    for page in filings.get('files', []):
        if found >= FILINGS_PER_BANK:
            break

        time.sleep(REQUEST_DELAY)
        logger.info('Consultando página adicional %s', page['name'])

        response = session.get(
            SUBMISSIONS_PAGE_URL.format(name=page['name']), timeout=30
        )
        response.raise_for_status()

        frame = filings_to_frame(response.json())
        frames.append(frame)
        found += (frame['form'] == FORM_TYPE).sum()

    return pd.concat(frames, ignore_index=True)


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
            `ticker`, `name`, `cik` y `fiscal_year` agregadas al inicio.
            `fiscal_year` es el año de `reportDate` (cierre del periodo
            informado), no el de `filingDate`: el 10-K del año fiscal 2025
            se presenta en 2026.
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
    selected.insert(
        3, 'fiscal_year', pd.to_datetime(selected['reportDate']).dt.year
    )

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
            logger.info('Ya existe %s, se omite', target.name)
            continue

        response = session.get(r.url, timeout=60)
        response.raise_for_status()

        target.write_bytes(response.content)
        logger.info('Descargado %s', target.name)

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
        logger.info('Consultando %s (CIK %s)', bank['ticker'], bank['cik'])

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

    logger.info('Tabla guardada en %s (%d filas)', csv_path, len(table))
    logger.info('Descargando documentos')

    download_filings(session, table)


# =============================================================================
def run() -> None:
    """Configura los logs y ejecuta main; es el punto de entrada del comando."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    main()


# =============================================================================
if __name__ == "__main__":
    run()
