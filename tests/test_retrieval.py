"""Pruebas de la búsqueda filtrada (embeddings falsos, sin modelo)."""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_analyst_agent.rag import retrieval as rt
from financial_analyst_agent.rag import vectorstore as vs

BANKS = [
    {'ticker': 'JPM', 'name': 'JPMorgan Chase', 'aliases': ['JPMorgan', 'JPM']},
    {'ticker': 'BAC', 'name': 'Bank of America', 'aliases': ['BofA', 'BAC']},
    {'ticker': 'C', 'name': 'Citigroup', 'aliases': ['Citi']},
    {'ticker': 'MS', 'name': 'Morgan Stanley', 'aliases': ['MS']},
]
PATTERNS = rt.bank_patterns(BANKS)


@pytest.mark.parametrize('question, tickers', [
    ("What drove JPMorgan Chase's net interest income?", ['JPM']),
    ('How did bank of america deposits change?', ['BAC']),
    ('Compare Citi and BofA capital ratios', ['BAC', 'C']),
    ('What does Morgan Stanley say about wealth management?', ['MS']),
    ('What risks do banks face?', []),
])
def test_find_tickers(question, tickers):
    assert rt.find_tickers(question, PATTERNS) == tickers


def test_morgan_stanley_is_not_found_inside_jpmorgan():
    assert rt.find_tickers('JPMorgan results in 2024', PATTERNS) == ['JPM']


def test_lowercase_ticker_is_not_a_bank():
    assert rt.find_tickers('Latency in ms and jpm units', PATTERNS) == []


def test_repo_config_has_aliases_for_every_bank():
    patterns = rt.bank_patterns()

    assert rt.find_tickers('Goldman, Truist, Citi, BofA, JPMorgan, Morgan Stanley', patterns) == [
        'BAC', 'C', 'GS', 'JPM', 'MS', 'TFC',
    ]


@pytest.mark.parametrize('question, years', [
    ('Net income in 2024?', [2024]),
    ('Compare FY2023 with 2025', [2023, 2025]),
    ('Deposits of $20245 million', []),
    ('No year here', []),
])
def test_find_years(question, years):
    assert rt.find_years(question) == years


def test_build_filter():
    assert rt.build_filter([], []) is None
    assert rt.build_filter(['GS'], []) == {'ticker': {'$in': ['GS']}}
    assert rt.build_filter(['GS'], [2025]) == {
        '$and': [{'ticker': {'$in': ['GS']}}, {'fiscal_year': {'$in': [2025]}}]
    }


def test_search_only_returns_bank_and_year_from_question(tmp_path):
    store = vs.get_vectorstore(tmp_path, DeterministicFakeEmbedding(size=8))
    question = 'How did Bank of America deposits change in 2023?'
    docs = [
        # El chunk de otro banco tiene el mismo texto que la pregunta: sin
        # filtro saldría primero.
        Document(question, metadata={'ticker': 'JPM', 'fiscal_year': 2023}),
        Document('Deposits fell.', metadata={'ticker': 'BAC', 'fiscal_year': 2023}),
        Document('Deposits rose.', metadata={'ticker': 'BAC', 'fiscal_year': 2024}),
    ]
    store.add_documents(docs, ids=['a', 'b', 'c'])

    results = rt.search(store, question, k=3, banks=BANKS)

    assert [d.page_content for d in results] == ['Deposits fell.']
