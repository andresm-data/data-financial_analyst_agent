"""Pruebas de la comprobación de búsqueda (embeddings falsos, sin modelo)."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_analyst_agent.rag import search_check as sc
from financial_analyst_agent.rag import vectorstore as vs

QUESTION = sc.Question('How did deposits change?', 'BAC', 2023)


def chunk(ticker, year, text='Deposits fell.', **overrides):
    meta = {key: 'x' for key in sc.REQUIRED_METADATA}
    meta.update(ticker=ticker, fiscal_year=year, chunk_index=0,
                chunk_id=f'{ticker}_{year}_mdna_0000', **overrides)
    return Document(text, metadata=meta)


def test_problems_is_empty_for_right_bank_and_year():
    assert sc.problems(chunk('BAC', 2023), QUESTION) == []


def test_problems_reports_wrong_bank_year_and_missing_metadata():
    issues = sc.problems(chunk('JPM', 2024, section=''), QUESTION)

    assert issues == ['falta section', 'banco JPM != BAC', 'año 2024 != 2023']


def test_check_counts_wrong_results(tmp_path, capsys):
    # El embedding falso depende solo del texto: la pregunta y el chunk con
    # el mismo texto quedan a distancia 0.
    store = vs.get_vectorstore(tmp_path, DeterministicFakeEmbedding(size=8))
    docs = [chunk('BAC', 2023, QUESTION.text), chunk('JPM', 2024, 'Other text.')]
    store.add_documents(docs, ids=[d.metadata['chunk_id'] for d in docs])

    assert sc.check(store, [QUESTION], k=2) == 1
    assert '1/2 resultados' in capsys.readouterr().out
