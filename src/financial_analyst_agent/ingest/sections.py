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
"""
import re
from dataclasses import dataclass
from typing import Literal

# =============================================================================
SECTIONS = {
    'risk_factors': 'Risk Factors',
    'mdna': "Management's Discussion and Analysis",
}

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
