"""Localiza las secciones Risk Factors (Item 1A) y MD&A (Item 7) en el texto de un 10-K.

Cada sección se describe con uno o más tramos (`Span`): un encabezado de inicio
y un encabezado de fin, ambos como expresión regular que debe coincidir con la
línea completa. La sección va desde la línea de inicio (incluida) hasta la de
fin (excluida).

Un mismo encabezado aparece varias veces: en el índice, en el glosario o en
remisiones. Para elegir el bueno:

1. Cada inicio se empareja con el primer fin que aparece después.
2. Si varios inicios comparten el mismo fin, se queda el más cercano al fin
   (`pick='last'`, por defecto) o el más lejano (`pick='first'`, para bancos
   que repiten el título de la sección como encabezado de página).
3. Entre los pares resultantes gana el más largo: el del índice ocupa pocas
   líneas y el de la sección real ocupa cientos.

La mayoría de los bancos usa los encabezados estándar "Item 1A" e "Item 7".
Los que no, siguen su índice de referencias cruzadas (ver `BANK_RULES`).

`main` guarda cada sección en data/processed/sections/ como
`{ticker}_{año}_{sección}.txt` junto a un `.json` con sus metadatos, e imprime
un resumen de todos los reportes.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from financial_analyst_agent.ingest.download_10k import (
    PROJECT_ROOT,
    RAW_10K_DIR,
    load_banks,
)
from financial_analyst_agent.ingest.html_to_text import TEXT_DIR

logger = logging.getLogger(__name__)

# =============================================================================
SECTIONS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'sections'

SECTIONS = {
    'risk_factors': 'Risk Factors',
    'mdna': "Management's Discussion and Analysis",
}

# Una sección más corta que esto es una entrada del índice o una remisión
# ("ver páginas 46–160"), no la sección real, y se trata como no encontrada.
MIN_SECTION_CHARS = 2_000

# Una línea más larga que esto es un párrafo, no un encabezado.
MAX_HEADING_CHARS = 150

APOS = "[’']"


# =============================================================================
@dataclass(frozen=True)
class Span:
    """Tramo de una sección entre un encabezado de inicio y uno de fin."""

    start: str
    end: str
    pick: Literal['first', 'last'] = 'last'


# =============================================================================
DEFAULT_RULES = {
    'risk_factors': [
        Span(start=r'item\s*1a\b.*', end=r'item\s*(1b|1c|2)\b.*'),
    ],
    'mdna': [
        Span(start=r'item\s*7(?![0-9a-z]).*', end=r'item\s*(7a|8)\b.*'),
    ],
}

BANK_RULES = {
    # JPM: "Item 7" solo remite a "Management's discussion and analysis"
    # (págs. 46–160), que termina antes del informe de control interno. El
    # título se repite en cada página, por eso se usa la primera aparición.
    'JPM': {
        'mdna': [
            Span(
                start=rf'management{APOS}s discussion and analysis',
                end=rf'management{APOS}s report on internal control over '
                    r'financial reporting',
                pick='first',
            ),
        ],
    },
    # C: índice de referencias cruzadas. 1A = "RISK FACTORS" (págs. 49–62);
    # 7 = MD&A (págs. 8–36) + "MANAGING GLOBAL RISK" (págs. 64–120).
    # Los títulos van en mayúsculas, así que se distinguen mayúsculas. La
    # sección que sigue a Risk Factors cambia de nombre según el año.
    'C': {
        'risk_factors': [
            Span(
                start=r'(?-i:RISK FACTORS)',
                end=r'(?-i:CLIMATE CHANGE AND NET ZERO'
                    r'|NET ZERO AND SUSTAINABILITY|SUSTAINABILITY)',
            ),
        ],
        'mdna': [
            Span(
                start=rf'(?-i:MANAGEMENT{APOS}S DISCUSSION AND ANALYSIS'
                      r'( OF FINANCIAL CONDITION)?)',
                end=r'(?-i:CAPITAL RESOURCES)',
            ),
            Span(
                start=r'(?-i:MANAGING GLOBAL RISK)',
                end=r'(?-i:SIGNIFICANT ACCOUNTING POLICIES AND'
                    r'( SIGNIFICANT ESTIMATES)?)',
            ),
        ],
    },
    # MS: índice por título, sin "Item". 1A = "Risk Factors" hasta
    # "Cybersecurity"; 7 = MD&A hasta "Quantitative and Qualitative
    # Disclosures about Risk" (7A).
    'MS': {
        'risk_factors': [
            Span(start=r'risk factors', end=r'cybersecurity'),
        ],
        'mdna': [
            Span(
                start=rf'management{APOS}s discussion and analysis of '
                      r'financial condition and results of operations',
                end=r'quantitative and qualitative disclosures about risk',
            ),
        ],
    },
}


# =============================================================================
def rules_for(ticker: str) -> dict[str, list[Span]]:
    """Reglas de un banco: las por defecto con las propias del banco encima."""
    return DEFAULT_RULES | BANK_RULES.get(ticker, {})


# =============================================================================
def heading_lines(lines: list[str], pattern: str) -> list[int]:
    """Índices de las líneas cortas que coinciden completas con `pattern`."""
    regex = re.compile(pattern, re.IGNORECASE)

    return [
        i for i, line in enumerate(lines)
        if len(line) <= MAX_HEADING_CHARS and regex.fullmatch(line)
    ]


# =============================================================================
def locate_span(lines: list[str], span: Span) -> tuple[int, int] | None:
    """Ubica un tramo en el texto según las reglas de este módulo.

    Args:
        lines (list[str]): Líneas del 10-K.
        span (Span): Encabezados de inicio y fin del tramo.

    Returns:
        tuple[int, int] | None: Línea de inicio (incluida) y de fin
            (excluida) del tramo más largo, o None si no hay ningún par.
    """
    starts = heading_lines(lines, span.start)
    ends = heading_lines(lines, span.end)

    pairs: dict[int, int] = {}

    for start in starts:
        end = next((e for e in ends if e > start), None)

        if end is None:
            continue

        if end not in pairs or span.pick == 'last':
            pairs[end] = start

    if not pairs:
        return None

    return max(
        ((start, end) for end, start in pairs.items()),
        key=lambda pair: pair[1] - pair[0],
    )


# =============================================================================
def locate_section(
    lines: list[str], spans: list[Span]
) -> list[tuple[int, int]] | None:
    """Ubica todos los tramos de una sección; None si falta alguno."""
    found = [locate_span(lines, span) for span in spans]

    if any(f is None for f in found):
        return None

    return found


# =============================================================================
def extract_sections(text: str, ticker: str) -> dict[str, dict | None]:
    """Extrae Risk Factors y MD&A del texto de un 10-K.

    Args:
        text (str): Texto limpio del 10-K (una línea por bloque).
        ticker (str): Ticker del banco, para aplicar sus reglas propias.

    Returns:
        dict[str, dict | None]: Por cada sección de `SECTIONS`, un dict con
            `text` y `ranges` (lista de pares de líneas inicio/fin, base 0),
            o None si no se encontró.
    """
    lines = text.splitlines()
    result = {}

    for section, spans in rules_for(ticker).items():
        ranges = locate_section(lines, spans)

        if ranges is None:
            result[section] = None
            continue

        result[section] = {
            'text': '\n'.join(
                '\n'.join(lines[start:end]) for start, end in ranges
            ) + '\n',
            'ranges': ranges,
        }

    return result


# =============================================================================
def parse_report_name(text_file: Path) -> dict:
    """Saca ticker, fecha de presentación y año fiscal del nombre del archivo.

    El nombre tiene la forma `{ticker}_{filingDate}_{doc}-{YYYYMMDD}`; el año
    fiscal es el de la fecha de cierre del documento, no el de presentación.

    Raises:
        ValueError: Si el nombre no tiene esa forma.
    """
    match = re.fullmatch(
        r'(?P<ticker>[A-Z]+)_(?P<filing_date>\d{4}-\d{2}-\d{2})_'
        r'(?P<doc>.+-(?P<year>\d{4})\d{4})',
        text_file.stem,
    )

    if match is None:
        raise ValueError(f'Nombre de 10-K inesperado: {text_file.name}')

    return {
        'ticker': match['ticker'],
        'filing_date': match['filing_date'],
        'fiscal_year': int(match['year']),
    }


# =============================================================================
def relative(path: Path) -> str:
    """Ruta relativa a la raíz del proyecto, con barras normales."""
    return path.relative_to(PROJECT_ROOT).as_posix()


# =============================================================================
def save_report_sections(text_file: Path, companies: dict[str, str]) -> list[dict]:
    """Extrae y guarda las secciones de un 10-K con su JSON de metadatos.

    Si una sección no se encuentra o tiene menos de `MIN_SECTION_CHARS`, se
    da por faltante y se borran los archivos que pudiera haber de una
    ejecución anterior para no dejar datos viejos.

    Args:
        text_file (Path): Texto limpio del 10-K.
        companies (dict[str, str]): Nombre de la empresa por ticker.

    Returns:
        list[dict]: Una fila por sección con `ticker`, `fiscal_year`,
            `section` y `chars` (0 si no se encontró).
    """
    report = parse_report_name(text_file)
    found = extract_sections(text_file.read_text(encoding='utf-8'), report['ticker'])
    rows = []

    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for section, title in SECTIONS.items():
        stem = f"{report['ticker']}_{report['fiscal_year']}_{section}"
        txt_path = SECTIONS_DIR / f'{stem}.txt'
        json_path = SECTIONS_DIR / f'{stem}.json'
        data = found[section]

        if data is not None and len(data['text']) < MIN_SECTION_CHARS:
            logger.warning(
                '%s en %s tiene %d caracteres; se descarta por corta',
                title, text_file.name, len(data['text']),
            )
            data = None

        if data is None:
            txt_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
            logger.warning('No se encontró %s en %s', title, text_file.name)
            rows.append({**report, 'section': section, 'chars': 0})
            continue

        metadata = {
            'company': companies[report['ticker']],
            'ticker': report['ticker'],
            'fiscal_year': report['fiscal_year'],
            'filing_date': report['filing_date'],
            'section': section,
            'section_title': title,
            'source_file': relative(RAW_10K_DIR / f'{text_file.stem}.htm'),
            'text_file': relative(text_file),
            'section_file': relative(txt_path),
            'line_ranges': [list(r) for r in data['ranges']],
            'chars': len(data['text']),
        }

        txt_path.write_text(data['text'], encoding='utf-8')
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        rows.append({**report, 'section': section, 'chars': metadata['chars']})

    return rows


# =============================================================================
def build_summary(rows: list[dict]) -> pd.DataFrame:
    """Tabla con una fila por reporte y los caracteres de cada sección.

    Returns:
        pd.DataFrame: Columnas `ticker`, `fiscal_year`, una por sección
            (caracteres, 0 si falta) y `complete` (True si están todas).
    """
    summary = (
        pd.DataFrame(rows)
        .pivot_table(
            index=['ticker', 'fiscal_year'], columns='section',
            values='chars', aggfunc='sum',
        )
        .reindex(columns=list(SECTIONS))
        .reset_index()
    )
    summary.columns.name = None
    summary['complete'] = (summary[list(SECTIONS)] > 0).all(axis=1)

    return summary


# =============================================================================
def main() -> None:
    """Extrae las secciones de todos los 10-K de data/processed/text/.

    Guarda los archivos en data/processed/sections/, escribe summary.csv e
    imprime el resumen. Los reportes a los que les falta alguna sección se
    avisan en el log; no detienen el pipeline.

    Raises:
        FileNotFoundError: Si no hay ningún texto de 10-K.
    """
    text_files = sorted(TEXT_DIR.glob('*.txt'))

    if not text_files:
        raise FileNotFoundError(f'No hay textos de 10-K en {TEXT_DIR}')

    companies = {bank['ticker']: bank['name'] for bank in load_banks()}
    rows = [
        row for text_file in text_files
        for row in save_report_sections(text_file, companies)
    ]

    summary = build_summary(rows)
    summary.to_csv(SECTIONS_DIR / 'summary.csv', index=False)

    print(summary.to_string(index=False))

    incomplete = summary[~summary['complete']]
    if incomplete.empty:
        logger.info('%d reportes con todas las secciones', len(summary))
    else:
        logger.warning(
            '%d de %d reportes sin todas las secciones: %s',
            len(incomplete), len(summary),
            ', '.join(f'{r.ticker} {r.fiscal_year}' for r in incomplete.itertuples()),
        )


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
