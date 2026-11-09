"""Pruebas de la colección de Chroma (embeddings falsos, sin red ni modelo)."""
import pytest
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


def make_chunks(n, text='Credit losses rose in quarter {i}.'):
    return [
        Document(text.format(i=i), metadata={'ticker': 'JPM', 'chunk_id': f'JPM_2024_mdna_{i:04d}'})
        for i in range(n)
    ]


def test_index_chunks_adds_in_batches(tmp_path, monkeypatch):
    store = vs.get_vectorstore(tmp_path, FAKE)
    calls = []
    original = store.add_documents
    monkeypatch.setattr(store, 'add_documents', lambda docs, ids: calls.append(len(ids)) or original(docs, ids=ids))

    stats = vs.index_chunks(store, make_chunks(250), batch_size=100)

    assert calls == [100, 100, 50]
    assert stats == vs.IndexStats(added=250, unchanged=0, deleted=0, total=250)


def test_index_chunks_twice_does_not_duplicate(tmp_path):
    chunks = make_chunks(30)
    vs.index_chunks(vs.get_vectorstore(tmp_path, FAKE), chunks)

    stats = vs.index_chunks(vs.get_vectorstore(tmp_path, FAKE), chunks)

    assert stats == vs.IndexStats(added=0, unchanged=30, deleted=0, total=30)


def test_index_chunks_updates_changed_and_deletes_stale(tmp_path):
    store = vs.get_vectorstore(tmp_path, FAKE)
    vs.index_chunks(store, make_chunks(5))

    new = make_chunks(3)
    new[0].page_content = 'Deposits fell.'
    stats = vs.index_chunks(store, new)

    assert stats == vs.IndexStats(added=1, unchanged=2, deleted=2, total=3)
    assert store.get(ids=['JPM_2024_mdna_0000'])['documents'] == ['Deposits fell.']


def test_index_chunks_rejects_duplicate_ids(tmp_path):
    chunks = make_chunks(2)
    chunks[1].metadata['chunk_id'] = chunks[0].metadata['chunk_id']

    with pytest.raises(ValueError):
        vs.index_chunks(vs.get_vectorstore(tmp_path, FAKE), chunks)
