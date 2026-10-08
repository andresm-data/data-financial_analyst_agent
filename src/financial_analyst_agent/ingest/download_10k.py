"""Descarga los 3 informes 10-K más recientes de cada banco desde SEC EDGAR."""
import os
import sys
import unicodedata
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv


# =============================================================================
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]

COMPANIES_FILE = PACKAGE_DIR / 'config' / 'companies.yaml'
ENV_FILE = PACKAGE_DIR / '.env'


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
