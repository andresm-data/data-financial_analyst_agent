"""Pruebas de la detección de títulos de apartado."""
import pytest

from financial_analyst_agent.rag import headings as hd


@pytest.mark.parametrize('line, next_line, expected', [
    ('ASSET & WEALTH MANAGEMENT', 'Net income was $5.4 billion.', True),
    ('Consumer and Small Business Banking', 'CSBB net income was $3.5 billion.', True),
    ('Net Interest Income and NIM', None, True),
    ('INTRODUCTION', None, True),
    ('Deposits', 'Deposits decreased $6.5 billion primarily due to higher customer spending.', True),
    ('Overview', 'This section describes the results of operations for the full year.', True),
    ('Deposits', None, False),                          # una palabra sin párrafo después
    ('Average', 'Balance', False),                      # trozo de encabezado de tabla
    ('Total', 'Total deposits 1,923,827 1,930,341', False),
    ('Deposits', 'see page 46.', False),                # lo que sigue no es un párrafo
    ('Net income was $5.4 billion, up 4%.', None, False),
    ('Revenue by line of business', None, False),       # no todas en mayúscula
    ('Percent of', None, False),                        # termina en preposición
    ('Year Ended December', None, False),               # encabezado de columnas
    ('Net Revenues', '$ in millions 2024 2023', False),  # seguido de una fila de tabla
    ('SUMMARY OF SELECTED FINANCIAL DATA', 'Net income 12,682 9,228', True),
    ('•Deposits', None, False),
    ('JPMorgan Chase & Co./2024 Form 10-K', None, False),
    ('Table of Contents', None, False),
])
def test_is_heading(line, next_line, expected):
    assert hd.is_heading(line, next_line) is expected


def test_find_headings_drops_page_headers():
    page = 'THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES\nSome text here.\n'
    text = page * 5 + 'Asset & Wealth Management\nNet revenues were higher.'

    assert [h for _, h in hd.find_headings(text)] == ['Asset & Wealth Management']


def test_heading_path_caps_style_keeps_parent():
    headings = [(0, 'CONSOLIDATED RESULTS'), (10, 'ASSET & WEALTH MANAGEMENT'), (20, 'Global Private Bank'),
                (30, 'CORPORATE')]

    assert hd.heading_path(headings, 5) == 'CONSOLIDATED RESULTS'
    assert hd.heading_path(headings, 15) == 'ASSET & WEALTH MANAGEMENT'
    assert hd.heading_path(headings, 25) == 'ASSET & WEALTH MANAGEMENT > Global Private Bank'


def test_heading_path_ignores_a_lone_caps_heading():
    headings = [(0, 'MSPBNA'), (10, 'Wealth Management'), (20, 'Net Revenues')]

    assert hd.heading_path(headings, 25) == 'Net Revenues'


def test_heading_path_title_style_keeps_last():
    headings = [(0, 'Market-Making Activities'), (10, 'Asset & Wealth Management')]

    assert hd.heading_path(headings, 15) == 'Asset & Wealth Management'
    assert hd.heading_path([], 15) == ''


def test_chunk_headings_follow_chunks_in_order():
    text = ('INTRODUCTION\nTotal assets were $4.0 trillion.\n'
            'ASSET & WEALTH MANAGEMENT\nTotal assets $ 255,385')
    chunks = ['INTRODUCTION\nTotal assets were $4.0 trillion.', 'ASSET & WEALTH MANAGEMENT\nTotal assets $ 255,385']

    assert hd.chunk_headings(text, chunks) == ['INTRODUCTION', 'ASSET & WEALTH MANAGEMENT']


def test_chunk_offsets_reuse_previous_when_not_found():
    assert hd.chunk_offsets('abc\ndef', ['def', 'zzz']) == [4, 4]
