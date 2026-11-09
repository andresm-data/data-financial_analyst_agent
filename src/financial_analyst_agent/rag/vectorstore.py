"""Colección persistente de Chroma con los chunks de los 10-K.

La base de datos vive en data/chroma/ (ignorada por git) y usa los mismos
embeddings bge-small de `financial_analyst_agent.embeddings`, así que se
puede volver a abrir sin reindexar.

`main` indexa data/processed/chunks/chunks.jsonl en lotes de `BATCH_SIZE`.
Es repetible: cada chunk se guarda con su `chunk_id` como ID, solo se
embeben los chunks nuevos o cambiados y se borran los que ya no existen.
"""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from financial_analyst_agent.embeddings import get_embeddings
from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT
from financial_analyst_agent.rag.chunking import CHUNKS_FILE, load_chunks

logger = logging.getLogger(__name__)

# =============================================================================
CHROMA_DIR = PROJECT_ROOT / 'data' / 'chroma'
COLLECTION_NAME = 'tenk_chunks'

# Los vectores de bge-small vienen normalizados: la distancia coseno es la
# métrica natural y da puntuaciones de relevancia entre 0 y 1.
COLLECTION_METADATA = {'hnsw:space': 'cosine'}

BATCH_SIZE = 100


# =============================================================================
@dataclass(frozen=True)
class IndexStats:
    """Resultado de una indexación."""

    added: int      # chunks nuevos o cambiados, embebidos en esta ejecución
    unchanged: int  # chunks que ya estaban iguales en la colección
    deleted: int    # IDs que ya no están en chunks.jsonl
    total: int      # chunks en la colección al terminar


# =============================================================================
def get_vectorstore(
    persist_dir: Path = CHROMA_DIR, embeddings: Embeddings | None = None
) -> Chroma:
    """Abre la colección de Chroma en `persist_dir`, creándola si no existe.

    Args:
        persist_dir (Path): Carpeta de la base de datos persistente.
        embeddings (Embeddings | None): Modelo de embeddings; por defecto,
            bge-small en CPU. Las pruebas pasan uno falso.

    Returns:
        Chroma: La colección lista para añadir o buscar chunks.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir),
        collection_metadata=COLLECTION_METADATA,
    )


# =============================================================================
def index_chunks(
    store: Chroma, chunks: list[Document], batch_size: int = BATCH_SIZE
) -> IndexStats:
    """Sincroniza la colección con `chunks`, añadiendo en lotes.

    Args:
        store (Chroma): Colección de destino.
        chunks (list[Document]): Chunks con `chunk_id` en la metadata.
        batch_size (int): Chunks por llamada a Chroma (y al modelo).

    Returns:
        IndexStats: Cuántos chunks se añadieron, se dejaron y se borraron.

    Raises:
        ValueError: Si hay `chunk_id` repetidos.
    """
    ids = [chunk.metadata['chunk_id'] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError('Hay chunk_id repetidos en los chunks')

    stored = store.get(include=['documents', 'metadatas'])
    existing = {
        id_: (text, metadata)
        for id_, text, metadata in zip(
            stored['ids'], stored['documents'], stored['metadatas']
        )
    }

    stale = sorted(existing.keys() - set(ids))
    if stale:
        store.delete(ids=stale)

    pending = [
        (id_, chunk) for id_, chunk in zip(ids, chunks)
        if existing.get(id_) != (chunk.page_content, chunk.metadata)
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        store.add_documents([chunk for _, chunk in batch], ids=[id_ for id_, _ in batch])
        logger.info('Indexados %d/%d chunks', start + len(batch), len(pending))

    return IndexStats(
        added=len(pending),
        unchanged=len(chunks) - len(pending),
        deleted=len(stale),
        total=store._collection.count(),
    )


# =============================================================================
def main() -> None:
    """Indexa chunks.jsonl en data/chroma/ e imprime el total y el tiempo.

    Raises:
        FileNotFoundError: Si no existe chunks.jsonl.
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f'No existe {CHUNKS_FILE}; ejecuta `chunks` antes')

    chunks = load_chunks()
    start = time.perf_counter()
    stats = index_chunks(get_vectorstore(), chunks)
    elapsed = time.perf_counter() - start

    print(
        f'{stats.total} chunks en la colección {COLLECTION_NAME!r} ({CHROMA_DIR}); '
        f'{stats.added} añadidos, {stats.unchanged} sin cambios, '
        f'{stats.deleted} borrados en {elapsed:.1f} s'
    )


# =============================================================================
def run() -> None:
    """Configura los logs y ejecuta main; es el punto de entrada del comando."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    main()


# =============================================================================
if __name__ == '__main__':
    run()
