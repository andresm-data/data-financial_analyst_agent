"""Mide la detección de tablas de `tables` contra los <table> del HTML original.

Referencia: las líneas de texto que salen de una tabla de datos del 10-K, es
decir, un <table> con al menos `MIN_NUMERIC_ROWS` filas (<tr>) con alguna celda
numérica. Así se descartan las tablas de maquetación (viñetas, notas).

Para cada línea de cada sección (ya limpia, como se divide en chunks):

- predicha: está dentro de un bloque de tabla de `split_blocks` (sin contar la
  frase de presentación, que a propósito viene del texto corrido);
- real: su texto aparece en una tabla de datos del mismo 10-K.

Imprime precisión y recall por banco y ejemplos de errores. Uso:
`poetry run table_eval`.
"""
import json
import random
import warnings
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT
from financial_analyst_agent.ingest.html_to_text import html_to_text
from financial_analyst_agent.ingest.sections import SECTIONS_DIR
from financial_analyst_agent.rag.chunking import section_text
from financial_analyst_agent.rag.tables import NUMBER_CELL, split_blocks

# =============================================================================
MIN_NUMERIC_ROWS = 2
EXAMPLES = 5


# =============================================================================
def is_numeric_row(tr) -> bool:
    """True si alguna celda del <tr> tiene una cifra."""
    return any(
        NUMBER_CELL.fullmatch(text) and any(ch.isdigit() for ch in text)
        for cell in tr.find_all(['td', 'th'])
        if (text := cell.get_text(' ', strip=True).replace('\xa0', ' '))
    )


# =============================================================================
def data_table_lines(html: str | bytes) -> set[str]:
    """Líneas de texto (como las deja `html_to_text`) de las tablas de datos."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, 'lxml')

    lines: set[str] = set()
    # Solo tablas sin otra tabla dentro, para no contar dos veces.
    for table in soup.find_all('table'):
        if table.find('table') is not None:
            continue
        if sum(is_numeric_row(tr) for tr in table.find_all('tr')) < MIN_NUMERIC_ROWS:
            continue
        lines.update(html_to_text(str(table)).splitlines())

    return lines


# =============================================================================
def predicted_table_lines(text: str) -> tuple[list[str], set[int]]:
    """Líneas del texto y los índices que `split_blocks` marca como tabla."""
    lines, predicted = [], set()

    for block in split_blocks(text):
        for k, line in enumerate(block.lines):
            if block.is_table and k >= block.intro:
                predicted.add(len(lines))
            lines.append(line)

    return lines, predicted


# =============================================================================
def evaluate() -> tuple[pd.DataFrame, dict[str, list]]:
    """Precisión y recall por banco, y líneas mal clasificadas por tipo."""
    truth_cache: dict[str, set[str]] = {}
    counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    errors: dict[str, list] = {'false_positive': [], 'false_negative': []}

    for metadata_file in sorted(SECTIONS_DIR.glob('*_*_*.json')):
        section = json.loads(metadata_file.read_text(encoding='utf-8'))
        source = section['source_file']
        if source not in truth_cache:
            truth_cache[source] = data_table_lines((PROJECT_ROOT / source).read_bytes())
        truth = truth_cache[source]

        lines, predicted = predicted_table_lines(section_text(section))
        bank = counts[section['ticker']]

        for i, line in enumerate(lines):
            is_pred, is_true = i in predicted, line in truth
            if is_pred and is_true:
                bank['tp'] += 1
            elif is_pred:
                bank['fp'] += 1
                errors['false_positive'].append((section['ticker'], line))
            elif is_true:
                bank['fn'] += 1
                errors['false_negative'].append((section['ticker'], line))

    table = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    table.loc['TOTAL'] = table.sum()
    table['precision'] = (table['tp'] / (table['tp'] + table['fp'])).round(3)
    table['recall'] = (table['tp'] / (table['tp'] + table['fn'])).round(3)

    return table.rename_axis('ticker').reset_index(), errors


# =============================================================================
def main() -> None:
    """Imprime la evaluación y algunos errores de cada tipo."""
    table, errors = evaluate()
    print(table.to_string(index=False))

    rng = random.Random(0)
    for kind, examples in errors.items():
        print(f'\n{kind} ({len(examples)}), ejemplos:')
        for ticker, line in rng.sample(examples, min(EXAMPLES, len(examples))):
            print(f'  {ticker:4} {line[:110]}')


# =============================================================================
if __name__ == '__main__':
    main()
