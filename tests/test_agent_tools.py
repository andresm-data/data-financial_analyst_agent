"""Pruebas de los nodos `retrieve_text` y `query_sql` sin Chroma ni modelo."""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from financial_analyst_agent.agent.tools import query_sql, retrieve_text

DOC = Document(
    page_content='Climate change could disrupt our businesses.',
    metadata={'company': 'Goldman Sachs', 'fiscal_year': 2024, 'section': 'risk_factors',
              'section_title': 'Risk Factors', 'heading': 'Climate',
              'source_file': 'data/raw/10k/GS.htm'},
)


def test_retrieve_text_keeps_company_year_section_and_file():
    update = retrieve_text({'question': 'q'}, retriever=RunnableLambda(lambda q: [DOC] * 7))
    assert len(update['sources']) == 5
    source = update['sources'][0]
    assert (source['company'], source['fiscal_year'], source['section'], source['source_file']) == (
        'Goldman Sachs', 2024, 'Risk Factors > Climate', 'data/raw/10k/GS.htm')
    assert source['label'] == 'Goldman Sachs | FY2024 | Risk Factors > Climate'
    assert 'error' not in update and 'answer' not in update


def test_retrieve_text_stores_error_instead_of_raising():
    def boom(question):
        raise RuntimeError('chroma down')

    update = retrieve_text({'question': 'q'}, retriever=RunnableLambda(boom))
    assert update['sources'] == [] and 'chroma down' in update['error']


def test_query_sql_exposes_sql_rows_and_error():
    ok = {'sql': 'SELECT 1', 'columns': ['x'], 'rows': [{'x': 1}], 'error': None, 'attempts': 1}
    assert query_sql({'question': 'q'}, tool=lambda q: ok) == {
        'sql': 'SELECT 1', 'rows': [{'x': 1}], 'error': None}

    bad = {'sql': 'DROP TABLE financials', 'columns': [], 'rows': [], 'error': 'UnsafeSQLError: no', 'attempts': 1}
    update = query_sql({'question': 'q'}, tool=lambda q: bad)
    assert update['rows'] == [] and update['sql'] == 'DROP TABLE financials'
    assert update['error'] == 'SQL error: UnsafeSQLError: no'


def test_query_sql_keeps_previous_rag_error():
    bad = {'sql': 'x', 'columns': [], 'rows': [], 'error': 'boom', 'attempts': 3}
    update = query_sql({'question': 'q', 'error': 'RAG error: down'}, tool=lambda q: bad)
    assert update['error'] == 'RAG error: down | SQL error: boom'
