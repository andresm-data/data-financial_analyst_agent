"""Genera data/processed/manifest.csv con el origen y la fecha de cada descarga.

Las cifras financieras cambian con el tiempo (reexpresiones, nuevos 10-K), así
que cada archivo descargado queda registrado con su URL y su fecha de descarga.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from financial_analyst_agent.ingest.download_10k import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    filing_path,
    load_banks,
)
from financial_analyst_agent.ingest.load_financials import (
    company_facts_url,
    facts_path,
)

logger = logging.getLogger(__name__)


# =============================================================================
FILINGS_CSV = PROCESSED_DIR / 'filings.csv'
MANIFEST_PATH = PROCESSED_DIR / 'manifest.csv'

COLUMNS = ['file', 'source_url', 'downloaded_at']


# =============================================================================
def file_timestamp(path: Path) -> str:
    """Fecha de última escritura del archivo en ISO 8601 (UTC, segundos).

    Los pasos de descarga nunca reescriben un archivo existente, así que
    esta fecha coincide con la de descarga.
    """
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return mtime.isoformat(timespec='seconds')


# =============================================================================
def downloaded_sources(
    filings: pd.DataFrame, banks: list[dict]
) -> list[tuple[Path, str]]:
    """Lista (ruta local, URL de origen) de cada archivo descargado.

    Args:
        filings (pd.DataFrame): Tabla de filings.csv con las columnas
            `ticker`, `filingDate`, `primaryDocument` y `url`.
        banks (list[dict]): Bancos de config/companies.yaml.

    Returns:
        list[tuple[Path, str]]: Primero los 10-K y luego los company facts.
    """
    sources = [
        (filing_path(r.ticker, r.filingDate, r.primaryDocument), r.url)
        for r in filings.itertuples(index=False)
    ]
    sources += [
        (facts_path(bank['ticker']), company_facts_url(bank['cik']))
        for bank in banks
    ]

    return sources


# =============================================================================
def previous_dates(manifest_path: Path) -> dict[tuple[str, str], str]:
    """Fechas ya registradas, por (archivo, URL), en un manifiesto anterior."""
    if not manifest_path.exists():
        return {}

    old = pd.read_csv(manifest_path, dtype=str)

    return {
        (r.file, r.source_url): r.downloaded_at
        for r in old.itertuples(index=False)
    }


# =============================================================================
def build_manifest(
    sources: list[tuple[Path, str]],
    previous: dict[tuple[str, str], str] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Arma la tabla del manifiesto.

    Si un archivo con la misma URL ya estaba en el manifiesto anterior, se
    conserva su fecha; si no, se usa la fecha de escritura del archivo.

    Args:
        sources (list[tuple[Path, str]]): Pares (ruta local, URL de origen).
        previous (dict | None): Fechas del manifiesto anterior.
        root (Path | None): Raíz para las rutas relativas; por defecto la
            raíz del proyecto.

    Returns:
        pd.DataFrame: Columnas `file`, `source_url` y `downloaded_at`.

    Raises:
        FileNotFoundError: Si falta algún archivo que debió descargarse.
    """
    previous = previous or {}
    root = root or PROJECT_ROOT
    missing = [str(path) for path, _ in sources if not path.exists()]

    if missing:
        raise FileNotFoundError(f'Faltan archivos descargados: {missing}')

    rows = []

    for path, url in sources:
        file = path.relative_to(root).as_posix()
        downloaded_at = previous.get((file, url)) or file_timestamp(path)
        rows.append((file, url, downloaded_at))

    return pd.DataFrame(rows, columns=COLUMNS)


# =============================================================================
def write_manifest(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Genera y guarda manifest.csv a partir de filings.csv y los bancos."""
    filings = pd.read_csv(FILINGS_CSV, dtype=str)
    sources = downloaded_sources(filings, load_banks())
    manifest = build_manifest(sources, previous_dates(manifest_path))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    logger.info(
        'Manifiesto guardado en %s (%d archivos)', manifest_path, len(manifest)
    )

    return manifest
