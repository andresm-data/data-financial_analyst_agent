"""Reordena los candidatos con un cross-encoder (BAAI/bge-reranker-base, en CPU).

Un cross-encoder lee la pregunta y el chunk juntos, así que juzga mejor la
relevancia que la similitud de embeddings, pero es más lento: por eso solo
se aplica a los ~30 candidatos de la búsqueda híbrida. La primera ejecución
descarga el modelo (~1 GB) a la caché de Hugging Face, fuera del repo.
"""
from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from financial_analyst_agent.rag.retrieval import K

# =============================================================================
RERANK_MODEL = 'BAAI/bge-reranker-base'

# Recibe pares (pregunta, texto) y devuelve una puntuación por par.
Scorer = Callable[[list[tuple[str, str]]], Sequence[float]]


# =============================================================================
@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Cross-encoder bge-reranker-base en CPU (se carga una sola vez)."""
    return CrossEncoder(RERANK_MODEL, device='cpu')


def rerank(question: str, docs: list[Document], scorer: Scorer, top_n: int = K) -> list[Document]:
    """Los `top_n` chunks más relevantes según `scorer`, con su `rerank_score`.

    Devuelve copias de los Document con la puntuación en la metadata, para
    poder usarla después (por ejemplo, como umbral de relevancia).
    """
    if not docs:
        return []

    scores = scorer([(question, doc.page_content) for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)[:top_n]

    return [
        Document(page_content=doc.page_content, metadata={**doc.metadata, 'rerank_score': float(score)})
        for score, doc in ranked
    ]
