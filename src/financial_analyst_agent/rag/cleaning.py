"""Limpia el texto de una sección antes de dividirlo en chunks.

Quita lo que el PDF/HTML repite en cada página y que no es contenido:

- Números de página sueltos ("74"). Solo cuentan los que siguen la secuencia
  de páginas con sus vecinos; un número suelto fuera de ella suele ser una
  celda de tabla y se conserva.
- Pies con el nombre de la empresa y el número de página ("Bank of America
  32", "26 Truist Financial Corporation").
- Pies del formulario con su año ("JPMorgan Chase & Co./2024 Form 10-K").
- Títulos de página sin cifras repetidos al menos `MIN_RUNNING_HEADER_REPEATS`
  veces que sean el título de la sección, contengan el nombre de la empresa o
  sean "Table of Contents".

Los .txt de data/processed/sections/ no cambian; la limpieza solo afecta al
texto que se divide.
"""
import re
from collections import Counter

# =============================================================================
PAGE_NUMBER = re.compile(r'\d{1,3}')
# Mayor salto entre dos números de página seguidos (páginas sin pie).
MAX_PAGE_STEP = 3
# Números sueltos vecinos que se miran a cada lado (puede haber una celda en medio).
PAGE_NEIGHBORS = 2

FORM_FOOTER = re.compile(r'.{0,40}\b(19|20)\d{2} Form 10-K', re.IGNORECASE)
COMPANY_FOOTER = re.compile(r'(?:\d{1,3} )?(?P<name>\D+?)(?: \d{1,3})?')
MAX_FOOTER_WORDS = 6

MIN_RUNNING_HEADER_REPEATS = 10
MAX_RUNNING_HEADER_CHARS = 80
TABLE_OF_CONTENTS = 'table of contents'


# =============================================================================
def normalize(text: str) -> str:
    """Minúsculas y apóstrofo recto, para comparar títulos."""
    return text.lower().replace('’', "'")


# =============================================================================
def page_number_lines(lines: list[str]) -> set[int]:
    """Índices de los números sueltos que son números de página.

    Un número suelto es de página si alguno de los `PAGE_NEIGHBORS` números
    sueltos anteriores es entre 1 y `MAX_PAGE_STEP` menor, o alguno de los
    siguientes es entre 1 y `MAX_PAGE_STEP` mayor.
    Así se aceptan varios tramos de páginas (una sección puede juntar partes
    separadas del 10-K) y se descartan las celdas sueltas de las tablas.
    """
    numbers = [(i, int(line)) for i, line in enumerate(lines) if PAGE_NUMBER.fullmatch(line)]

    def follows(a: int, b: int) -> bool:
        return 1 <= b - a <= MAX_PAGE_STEP

    return {
        i for k, (i, value) in enumerate(numbers)
        if any(follows(prev, value) for _, prev in numbers[max(0, k - PAGE_NEIGHBORS):k])
        or any(follows(value, nxt) for _, nxt in numbers[k + 1:k + 1 + PAGE_NEIGHBORS])
    }


# =============================================================================
def is_company_footer(line: str, company: str) -> bool:
    """True para "Bank of America 32" o "26 Truist Financial Corporation"."""
    match = COMPANY_FOOTER.fullmatch(line)

    return (
        match is not None
        and match['name'] != line
        and len(line.split()) <= MAX_FOOTER_WORDS
        and normalize(company) in normalize(match['name'])
    )


# =============================================================================
def running_headers(lines: list[str], company: str, section_title: str) -> set[str]:
    """Títulos de página que se repiten a lo largo de la sección."""
    counts = Counter(
        line for line in lines
        if len(line) <= MAX_RUNNING_HEADER_CHARS and not re.search(r'\d', line)
    )

    return {
        line for line, n in counts.items()
        if n >= MIN_RUNNING_HEADER_REPEATS and (
            normalize(line) in (normalize(section_title), TABLE_OF_CONTENTS)
            or normalize(company) in normalize(line)
        )
    }


# =============================================================================
def page_furniture(lines: list[str], company: str, section_title: str) -> set[int]:
    """Índices de las líneas que son pies, números o títulos de página."""
    headers = running_headers(lines, company, section_title)

    return page_number_lines(lines) | {
        i for i, line in enumerate(lines)
        if FORM_FOOTER.fullmatch(line)
        or is_company_footer(line, company)
        or line in headers
    }


# =============================================================================
def clean_section_text(text: str, company: str, section_title: str) -> str:
    """Texto de la sección sin pies, números ni títulos de página."""
    lines = text.splitlines()
    furniture = page_furniture(lines, company, section_title)

    return '\n'.join(line for i, line in enumerate(lines) if i not in furniture) + '\n'
