"""Detecta las tablas en el texto de un 10-K para no cortarlas por la mitad.

`html_to_text` deja cada fila de tabla en una línea, con las celdas separadas
por espacios, por ejemplo:

    Table 1 Summary Income Statement and Selected Financial Data
    (Dollars in millions) 2024 2023
    Net interest income $ 56,060 $ 56,931
    Per common share information
    Earnings $ 3.25 $ 3.10

Una tabla es una racha de líneas donde hay al menos `MIN_TABLE_ROWS` filas
(líneas cortas que terminan en una cifra) mezcladas con etiquetas (líneas
cortas que no son frases, como "Per common share information"). Antes de la
primera fila se suman hasta `MAX_TITLE_LINES` etiquetas como título.

El texto debe llegar sin pies ni títulos de página (ver `cleaning`), que si no
se cuelan como título de la tabla.
"""
import re
from dataclasses import dataclass

# =============================================================================
# Una celda numérica: 1,234  (56)  $  3.25  %  —  n/a  NM  (1) ...
NUMBER_CELL = re.compile(
    r'[($]*-?\$?[\d,]*\.?\d+\)?%?\)?|[—–-]+|\$|%|\)|bps|n/?[am]|NM|x',
    re.IGNORECASE,
)

MAX_ROW_CHARS = 200
MAX_LABEL_CHARS = 150
MIN_TABLE_ROWS = 2
MAX_TITLE_LINES = 3

SENTENCE_END = ('.', '?', '!')


# =============================================================================
def is_row(line: str) -> bool:
    """True si la línea parece una fila de tabla: corta y termina en cifra."""
    tokens = line.split()

    return (
        bool(tokens)
        and len(line) <= MAX_ROW_CHARS
        and not line.endswith(SENTENCE_END)
        and NUMBER_CELL.fullmatch(tokens[-1]) is not None
    )


# =============================================================================
def is_label(line: str) -> bool:
    """True si la línea es corta y no es una frase (título o etiqueta)."""
    return bool(line) and len(line) <= MAX_LABEL_CHARS and not line.endswith(SENTENCE_END)


# =============================================================================
@dataclass(frozen=True)
class Block:
    """Tramo de líneas consecutivas: texto corrido o una tabla."""

    lines: list[str]
    is_table: bool

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


# =============================================================================
def table_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Rangos (inicio incluido, fin excluido) de las tablas del texto."""
    row = [is_row(line) for line in lines]
    label = [is_label(line) for line in lines]
    ranges = []
    i = 0

    while i < len(lines):
        if not row[i]:
            i += 1
            continue

        # Avanza por filas y etiquetas; la tabla termina en la última fila.
        end = i
        rows = 0
        j = i
        while j < len(lines) and (row[j] or label[j]):
            if row[j]:
                rows += 1
                end = j + 1
            j += 1

        if rows >= MIN_TABLE_ROWS:
            start = i
            prev_end = ranges[-1][1] if ranges else 0
            while start > prev_end and i - start < MAX_TITLE_LINES and label[start - 1]:
                start -= 1
            ranges.append((start, end))

        i = max(end, i + 1)

    return ranges


# =============================================================================
def split_blocks(text: str) -> list[Block]:
    """Divide el texto en bloques alternos de texto corrido y tablas."""
    lines = text.splitlines()
    blocks = []
    pos = 0

    for start, end in table_ranges(lines):
        if start > pos:
            blocks.append(Block(lines[pos:start], is_table=False))
        blocks.append(Block(lines[start:end], is_table=True))
        pos = end

    if pos < len(lines):
        blocks.append(Block(lines[pos:], is_table=False))

    return blocks


# =============================================================================
def table_header(block: Block, max_chars: int) -> list[str]:
    """Título y encabezado de columnas: las líneas hasta la primera fila.

    Si no caben en `max_chars`, se quitan líneas del final; si ni la primera
    cabe, se recorta.
    """
    first_row = next(i for i, line in enumerate(block.lines) if is_row(line))
    header = block.lines[:first_row + 1]

    while len(header) > 1 and len('\n'.join(header)) > max_chars:
        header.pop()

    if len(header[0]) > max_chars:
        header = [header[0][:max_chars]]

    return header
