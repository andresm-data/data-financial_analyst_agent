"""Colección persistente de Chroma con los chunks de los 10-K.

La base de datos vive en data/chroma/ (ignorada por git) y usa los mismos
embeddings bge-small de `financial_analyst_agent.embeddings`, así que se
puede volver a abrir sin reindexar.
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from financial_analyst_agent.embeddings import get_embeddings
from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT

# =============================================================================
CHROMA_DIR = PROJECT_ROOT / 'data' / 'chroma'
COLLECTION_NAME = 'tenk_chunks'

# Los vectores de bge-small vienen normalizados: la distancia coseno es la
# métrica natural y da puntuaciones de relevancia entre 0 y 1.
COLLECTION_METADATA = {'hnsw:space': 'cosine'}


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
