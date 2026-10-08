"""Descarga los 3 informes 10-K más recientes de cada banco desde SEC EDGAR."""
from pathlib import Path

import yaml


# =============================================================================
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parents[1]

COMPANIES_FILE = PACKAGE_DIR / 'config' / 'companies.yaml'


# =============================================================================
def load_banks() -> list[dict]:
    """Carga la lista de bancos a analizar desde config/companies.yaml.

    Returns:
        list[dict]: Un diccionario por banco con las claves ``ticker``,
            ``name``, ``info`` y ``cik`` (CIK como texto de 10 dígitos).
    """
    with COMPANIES_FILE.open(encoding='utf-8') as f:
        return yaml.safe_load(f)['banks']
