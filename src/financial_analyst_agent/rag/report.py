"""Informe de calidad de los chunks guardados en data/processed/chunks/.

Mide, por banco y en total:

- `mid_table`: chunks cuya primera línea es una fila interior de una tabla
  (ni su inicio ni un encabezado repetido con "(continued)").
- `lowercase`: chunks que empiezan en minúscula, casi siempre a mitad de frase.
- `short`: chunks con menos de `SHORT_CHUNK_CHARS` caracteres.
- `continued`: chunks que son continuación de una tabla larga.
- `max_chars`: tamaño del chunk más largo.

También lista los encabezados de continuación más repetidos, para detectar
títulos o pies de página que se cuelan como encabezado de tabla.
"""
import json
from collections import Counter

import pandas as pd
from langchain_core.documents import Document

from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT
from financial_analyst_agent.rag.chunking import CONTINUED, load_chunks, section_text
from financial_analyst_agent.rag.tables import split_blocks

# =============================================================================
SHORT_CHUNK_CHARS = 200
TOP_HEADERS = 10


# =============================================================================
def first_line(chunk: Document) -> str:
    return chunk.page_content.split('\n', 1)[0]


# =============================================================================
def mid_table_ids(chunks: list[Document], text: str) -> set[str]:
    """IDs de los chunks de una sección que empiezan a mitad de una tabla.

    Una línea repetida en el texto es ambigua; solo cuenta como mitad de
    tabla si nunca aparece como inicio de tabla ni en texto corrido.
    """
    blocks = split_blocks(text)
    tables = [b for b in blocks if b.is_table]
    starts = {b.lines[0] for b in tables}
    inner = {line for b in tables for line in b.lines[1:]}
    prose = {line for b in blocks if not b.is_table for line in b.lines}

    return {
        c.metadata['chunk_id'] for c in chunks
        if (line := first_line(c)) in inner
        and line not in starts and line not in prose
        and not line.endswith(CONTINUED)
    }


# =============================================================================
def quality_table(chunks: list[Document]) -> pd.DataFrame:
    """Métricas de calidad por banco, con una fila TOTAL."""
    by_section: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.metadata['section_file'], []).append(chunk)

    mid = set()
    for section_file, section_chunks in by_section.items():
        metadata = json.loads(
            (PROJECT_ROOT / section_file).with_suffix('.json').read_text(encoding='utf-8')
        )
        mid |= mid_table_ids(section_chunks, section_text(metadata))

    rows = pd.DataFrame([
        {
            'ticker': c.metadata['ticker'],
            'chunks': 1,
            'mid_table': c.metadata['chunk_id'] in mid,
            'lowercase': c.page_content[:1].islower(),
            'short': len(c.page_content) < SHORT_CHUNK_CHARS,
            'continued': first_line(c).endswith(CONTINUED),
            'max_chars': len(c.page_content),
        }
        for c in chunks
    ])
    aggs = {col: 'sum' for col in ['chunks', 'mid_table', 'lowercase', 'short', 'continued']}
    aggs['max_chars'] = 'max'

    table = rows.groupby('ticker').agg(aggs)
    table.loc['TOTAL'] = rows.agg(aggs)

    return table.astype(int).reset_index()


# =============================================================================
def top_continued_headers(chunks: list[Document], n: int = TOP_HEADERS) -> list[tuple]:
    """Encabezados de continuación más repetidos: (ticker, encabezado, veces)."""
    counts = Counter(
        (c.metadata['ticker'], first_line(c).removesuffix(CONTINUED))
        for c in chunks if first_line(c).endswith(CONTINUED)
    )

    return [(ticker, header, n) for (ticker, header), n in counts.most_common(n)]


# =============================================================================
def main() -> None:
    """Imprime el informe de calidad de chunks.jsonl."""
    chunks = load_chunks()
    table = quality_table(chunks)
    total = table.set_index('ticker').loc['TOTAL']

    print(table.to_string(index=False))
    print(
        f"\nmid_table {total['mid_table'] / total['chunks']:.1%} | "
        f"lowercase {total['lowercase'] / total['chunks']:.1%} | "
        f"short {total['short'] / total['chunks']:.1%}"
    )
    print('\nEncabezados de continuación más repetidos:')
    for ticker, header, n in top_continued_headers(chunks):
        print(f'  {ticker:4} {n:3}  {header[:80]}')


# =============================================================================
if __name__ == '__main__':
    main()
