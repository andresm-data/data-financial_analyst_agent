"""Convierte los 10-K HTML de data/raw/10k/ a texto limpio en data/processed/text/."""
import logging
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT, RAW_10K_DIR

logger = logging.getLogger(__name__)


# =============================================================================
TEXT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'text'

# Etiquetas que no aportan texto legible. `ix:header` es el bloque oculto de
# XBRL en línea (contextos y datos de la portada que no se muestran).
DROP_TAGS = ['script', 'style', 'ix:header']

# Etiquetas que en el navegador empiezan una línea nueva.
BLOCK_TAGS = [
    'p', 'div', 'br', 'tr', 'li', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
]
CELL_TAGS = ['td', 'th']


# =============================================================================
def html_to_text(html: str | bytes) -> str:
    """Extrae el texto visible de un documento HTML.

    Elimina `<script>`, `<style>` y el encabezado oculto de XBRL, conserva un
    salto de línea por cada bloque (párrafo, div, fila de tabla) y separa las
    celdas de una tabla con espacios.

    Args:
        html (str | bytes): Contenido del documento HTML.

    Returns:
        str: Texto con espacios normalizados, sin líneas vacías y terminado
            en salto de línea.
    """
    # Los 10-K son XHTML (XBRL en línea). Se leen con el parser HTML a
    # propósito: tolera marcado imperfecto y entidades HTML como &nbsp;.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, 'lxml')

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()

    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before('\n')
        tag.insert_after('\n')

    for tag in soup.find_all(CELL_TAGS):
        tag.insert_after(' ')

    text = soup.get_text().replace('\xa0', ' ')
    lines = (re.sub(r'[ \t\r\f\v]+', ' ', line).strip() for line in text.split('\n'))

    return '\n'.join(line for line in lines if line) + '\n'


# =============================================================================
def text_path(raw_path: Path) -> Path:
    """Ruta del .txt de un 10-K: mismo nombre que el HTML con extensión .txt."""
    return TEXT_DIR / f'{raw_path.stem}.txt'


# =============================================================================
def convert_file(raw_path: Path) -> Path:
    """Convierte un 10-K a texto si el .txt no existe o es más viejo que el HTML.

    Args:
        raw_path (Path): Ruta del documento HTML.

    Returns:
        Path: Ruta del .txt generado (o del existente si estaba al día).
    """
    target = text_path(raw_path)

    if target.exists() and target.stat().st_mtime >= raw_path.stat().st_mtime:
        logger.info('Ya existe %s, se omite', target.name)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_to_text(raw_path.read_bytes()), encoding='utf-8')
    logger.info('Convertido %s', target.name)

    return target


# =============================================================================
def main() -> None:
    """Convierte a texto todos los 10-K de data/raw/10k/.

    Raises:
        FileNotFoundError: Si no hay ningún 10-K descargado.
    """
    raw_files = sorted(RAW_10K_DIR.glob('*.htm'))

    if not raw_files:
        raise FileNotFoundError(f'No hay 10-K en {RAW_10K_DIR}')

    for raw_path in raw_files:
        convert_file(raw_path)

    logger.info('%d 10-K convertidos a texto en %s', len(raw_files), TEXT_DIR)


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
