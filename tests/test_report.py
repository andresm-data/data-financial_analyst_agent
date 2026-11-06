"""Pruebas del informe de calidad de chunks (sin datos descargados)."""
from langchain_core.documents import Document

from financial_analyst_agent.rag import report

TEXT = '\n'.join([
    'Net income rose because revenue grew and expenses were flat.',
    'Table 1 Summary',
    '(Dollars in millions) 2024 2023',
    'Net interest income $ 56,060 $ 56,931',
    'Earnings $ 3.25 $ 3.10',
    'Revenue growth was strong across all segments this year.',
])


def chunk(chunk_id, text):
    return Document(page_content=text, metadata={'chunk_id': chunk_id})


def test_mid_table_ids_flags_only_inner_rows():
    chunks = [
        chunk('start', 'Table 1 Summary\n(Dollars in millions) 2024 2023'),
        chunk('mid', 'Earnings $ 3.25 $ 3.10'),
        chunk('continued', 'Table 1 Summary (continued)\nEarnings $ 3.25 $ 3.10'),
        chunk('prose', 'Revenue growth was strong across all segments this year.'),
    ]

    assert report.mid_table_ids(chunks, TEXT) == {'mid'}
