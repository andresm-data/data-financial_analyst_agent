"""Pruebas del prompt estricto del RAG."""
from langchain_core.documents import Document

from financial_analyst_agent.rag import prompt as pr

JPM = Document(
    page_content='Net interest income was $92.6 billion.',
    metadata={'company': 'JPMorgan Chase', 'fiscal_year': 2024, 'section': 'mdna',
              'section_title': "Management's Discussion and Analysis"},
)
BAC = Document(
    page_content='Deposits decreased.',
    metadata={'company': 'Bank of America', 'fiscal_year': 2023, 'section': 'risk_factors'},
)


def test_prompt_variables():
    assert set(pr.RAG_PROMPT.input_variables) == {'context', 'question'}


def test_prompt_demands_exact_no_answer():
    messages = pr.RAG_PROMPT.format_messages(context='ctx', question='q?')

    assert pr.NO_ANSWER == 'I do not know'
    assert 'ONLY' in messages[0].content
    assert 'reply exactly "I do not know" and nothing else' in messages[0].content
    assert 'Context:\nctx' in messages[1].content
    assert 'Question: q?' in messages[1].content


def test_source_label_uses_title_or_section():
    assert pr.source_label(JPM) == "JPMorgan Chase | FY2024 | Management's Discussion and Analysis"
    assert pr.source_label(BAC) == 'Bank of America | FY2023 | risk_factors'


def test_format_docs_numbers_sources_in_order():
    text = pr.format_docs([JPM, BAC])

    assert text.startswith('[1] JPMorgan Chase | FY2024')
    assert text.index('$92.6 billion') < text.index('[2] Bank of America | FY2023')
    assert text.endswith('Deposits decreased.')


def test_format_docs_empty():
    assert pr.format_docs([]) == ''
