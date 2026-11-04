"""Reconstruye todos los datos del proyecto ejecutando cada paso de ingesta.

Orden:
1. download_10k: lista de 10-K (data/processed/filings.csv) y documentos
   (data/raw/10k/).
2. html_to_text: un .txt limpio por 10-K en data/processed/text/.
3. sections: Risk Factors y MD&A con su JSON en data/processed/sections/.
4. load_financials: company facts (data/raw/facts/) y tabla `financials`
   en data/processed/finance.duckdb.
5. manifest: data/processed/manifest.csv con URL y fecha de cada descarga.
"""
import logging

from financial_analyst_agent.ingest import (
    download_10k,
    html_to_text,
    load_financials,
    manifest,
    sections,
)

logger = logging.getLogger(__name__)


# =============================================================================
STEPS = [
    ('download_10k', download_10k.main),
    ('html_to_text', html_to_text.main),
    ('sections', sections.main),
    ('load_financials', load_financials.main),
    ('manifest', manifest.write_manifest),
]


# =============================================================================
def main() -> None:
    """Ejecuta los pasos de ingesta en orden; si uno falla, se detiene."""
    for name, step in STEPS:
        logger.info('=== Paso: %s ===', name)
        step()

    logger.info('Datos reconstruidos')


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
