"""Títulos de apartado de las secciones de los 10-K, para dar contexto a los chunks.

Un chunk de tabla como "Total assets $ 255,385" no dice si es de toda la firma
o de un segmento; el título bajo el que está ("ASSET & WEALTH MANAGEMENT") sí.
Aquí se detectan esos títulos y se busca cuál corresponde a cada chunk.

Los 10-K usan dos estilos: títulos en mayúsculas (JPM, Citi) y títulos cortos
con cada palabra en mayúscula (GS, BAC, MS, Truist). Se descartan los
encabezados de página (líneas que se repiten muchas veces) y los de columnas
de tabla (seguidos de una fila con cifras o con nombres de meses).
"""
import re
from collections import Counter

from financial_analyst_agent.rag.tables import is_row

# =============================================================================
MIN_HEADING_CHARS = 4
MAX_HEADING_CHARS = 80
MIN_CAPS_HEADING_CHARS = 6   # un título de una sola palabra debe ir en mayúsculas
MAX_REPEATS = 5              # más repeticiones = encabezado de página
MIN_CAPS_HEADINGS = 3        # títulos en mayúsculas para usarlos como nivel superior

SMALL_WORDS = frozenset('a an and at by for from in of on or the to versus vs with'.split())
WORD = re.compile(r"[A-Za-z][A-Za-z’'.-]*")
MONTHS = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
    re.IGNORECASE,
)
SEPARATOR = ' > '


# =============================================================================
def is_heading(line: str, next_line: str | None = None) -> bool:
    """True si la línea parece un título de apartado.

    Corta, sin cifras ni puntuación final, con todas las palabras en mayúscula
    (salvo artículos y preposiciones). Si no está toda en mayúsculas, además
    no debe ir seguida de una fila de tabla.
    """
    line = line.strip()
    if not MIN_HEADING_CHARS <= len(line) <= MAX_HEADING_CHARS:
        return False
    if line.endswith(('.', ',', ':', ';')) or line.startswith(('•', '–', '-', '(')):
        return False
    if re.search(r'\d', line) or MONTHS.search(line) or 'table of contents' in line.lower():
        return False

    words = WORD.findall(line)
    if not words or not words[0][0].isupper() or not words[-1][0].isupper():
        return False
    if not all(word[0].isupper() or word.lower() in SMALL_WORDS for word in words):
        return False
    if len(words) < 2 and not (line.isupper() and len(line) >= MIN_CAPS_HEADING_CHARS):
        return False

    # Una línea en tipo título seguida de cifras suele ser un encabezado de
    # columnas; los títulos en mayúsculas sí pueden ir justo antes de una tabla.
    return line.isupper() or next_line is None or not is_row(next_line.strip())


def find_headings(text: str) -> list[tuple[int, str]]:
    """Títulos del texto con la posición (carácter) donde empiezan."""
    lines = text.split('\n')
    found = []
    offset = 0
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if is_heading(line, next_line):
            found.append((offset, line.strip()))
        offset += len(line) + 1

    counts = Counter(heading for _, heading in found)
    return [(offset, heading) for offset, heading in found if counts[heading] < MAX_REPEATS]


# =============================================================================
def heading_path(headings: list[tuple[int, str]], offset: int) -> str:
    """Títulos que rigen la posición `offset`, del más general al más concreto.

    Si el documento usa títulos en mayúsculas (al menos `MIN_CAPS_HEADINGS`),
    son el último en mayúsculas y el último normal que vino después; si no,
    solo el último título (el anterior suele ser un apartado hermano).
    """
    before = [heading for start, heading in headings if start <= offset]
    if not before:
        return ''

    caps_style = sum(heading.isupper() for _, heading in headings) >= MIN_CAPS_HEADINGS
    caps = [i for i, heading in enumerate(before) if heading.isupper()]
    if caps_style and caps:
        path = [before[caps[-1]], *before[caps[-1] + 1:][-1:]]
    else:
        path = before[-1:]

    return SEPARATOR.join(dict.fromkeys(path))


def chunk_offsets(text: str, chunks: list[str]) -> list[int]:
    """Posición aproximada de cada chunk en el texto.

    Se busca la línea más larga del chunk a partir de la posición del chunk
    anterior (los chunks van en orden). Si no aparece (por ejemplo, porque es
    un encabezado de tabla repetido), se usa la posición del anterior.
    """
    offsets = []
    cursor = 0
    for chunk in chunks:
        probe = max(chunk.split('\n'), key=len)
        found = text.find(probe, cursor)
        if found == -1:
            found = text.find(probe)
        if found != -1:
            cursor = found
        offsets.append(cursor)

    return offsets


def chunk_headings(text: str, chunks: list[str]) -> list[str]:
    """Título de apartado de cada chunk ('' si no hay ninguno antes)."""
    headings = find_headings(text)
    return [heading_path(headings, offset) for offset in chunk_offsets(text, chunks)]
