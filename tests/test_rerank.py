"""Pruebas del reranker (puntuaciones falsas, sin descargar el modelo)."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_analyst_agent.rag import chain as ch
from financial_analyst_agent.rag import rerank as rr
from financial_analyst_agent.rag import vectorstore as vs


def doc(chunk_id, text):
    return Document(page_content=text, metadata={'chunk_id': chunk_id, 'ticker': 'JPM', 'fiscal_year': 2024})


def keyword_scorer(word):
    """Puntúa 1 los textos que contienen `word` y 0 el resto."""
    return lambda pairs: [float(word in text) for _, text in pairs]


def test_rerank_orders_by_score_and_keeps_top_n():
    docs = [doc('a', 'deposits fell'), doc('b', 'CET1 ratio 13.6%'), doc('c', 'CET1 buffer')]

    ranked = rr.rerank('CET1?', docs, keyword_scorer('CET1'), top_n=2)

    assert [d.metadata['chunk_id'] for d in ranked] == ['b', 'c']
    assert [d.metadata['rerank_score'] for d in ranked] == [1.0, 1.0]
    assert 'rerank_score' not in docs[1].metadata


def test_rerank_empty():
    assert rr.rerank('q', [], keyword_scorer('x')) == []


def test_retriever_with_scorer_uses_hybrid_candidates(tmp_path):
    store = vs.get_vectorstore(tmp_path, DeterministicFakeEmbedding(size=8))
    docs = [doc(f'n{i}', f'noise text number {i}') for i in range(6)] + [doc('nii', 'Net interest income rose')]
    store.add_documents(docs, ids=[d.metadata['chunk_id'] for d in docs])

    retriever = ch.get_retriever(store, k=2, scorer=keyword_scorer('interest'), candidates=7)
    result = retriever.invoke('JPMorgan net interest income in 2024')

    assert len(result) == 2
    assert result[0].metadata['chunk_id'] == 'nii'
