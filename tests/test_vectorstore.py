"""Pruebas de la colección de Chroma (embeddings falsos, sin red ni modelo)."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from financial_analyst_agent.rag import vectorstore as vs

FAKE = DeterministicFakeEmbedding(size=8)


def test_collection_persists_and_reopens(tmp_path):
    store = vs.get_vectorstore(tmp_path, FAKE)
    store.add_documents(
        [Document('Net interest income rose.', metadata={'ticker': 'BAC'})],
        ids=['BAC_2024_mdna_0000'],
    )

    reopened = vs.get_vectorstore(tmp_path, FAKE)
    stored = reopened.get(include=['metadatas'])

    assert stored['ids'] == ['BAC_2024_mdna_0000']
    assert stored['metadatas'] == [{'ticker': 'BAC'}]


def test_collection_uses_cosine_distance(tmp_path):
    store = vs.get_vectorstore(tmp_path, FAKE)

    assert store._collection.metadata['hnsw:space'] == 'cosine'
