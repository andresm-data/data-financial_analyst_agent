"""Cadena RAG con LCEL: retriever de Chroma -> prompt -> llama3.2:3b -> texto.

La cadena devuelve la respuesta junto con los chunks recuperados, para
poder citar después la empresa, el año y la sección de cada fuente.

El retriever por defecto (`default_retriever`) hace búsqueda híbrida
(vectores + BM25) con 30 candidatos y los reordena con bge-reranker.
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.rag.prompt import NO_ANSWER, RAG_PROMPT, format_docs, source_label
from financial_analyst_agent.rag.rerank import Scorer, get_reranker, rerank
from financial_analyst_agent.rag.retrieval import CANDIDATES, K, hybrid_search, search
from financial_analyst_agent.rag.vectorstore import get_vectorstore


# =============================================================================
def get_retriever(
    store: Chroma, k: int = K, scorer: Scorer | None = None, candidates: int = CANDIDATES
) -> Runnable:
    """Retriever de Chroma con `k` resultados, filtrado por banco y año.

    Sin `scorer`, es la búsqueda vectorial de `retrieval.search`. Con
    `scorer`, junta `candidates` chunks de la búsqueda híbrida (vectores +
    BM25) y se queda con los `k` que mejor puntúa el reranker. En los dos
    casos se filtra por el banco y el año que nombra la pregunta.
    """
    if scorer is None:
        return RunnableLambda(lambda question: search(store, question, k=k), name='chroma_retriever')

    return RunnableLambda(
        lambda question: rerank(question, hybrid_search(store, question, k=candidates), scorer, top_n=k),
        name='hybrid_rerank_retriever',
    )


def default_retriever(store: Chroma) -> Runnable:
    """Búsqueda híbrida con 30 candidatos reordenados por bge-reranker (top 5)."""
    return get_retriever(store, scorer=get_reranker().predict)


# =============================================================================
def build_chain(retriever: Runnable, llm: BaseChatModel) -> Runnable:
    """Une retriever, prompt, modelo y parser con LCEL.

    Recibe la pregunta (str) y devuelve un dict con `question`, `context`
    (los Document recuperados) y `answer` (str).
    """
    answer = (
        RunnableLambda(lambda x: {'context': format_docs(x['context']), 'question': x['question']})
        | RAG_PROMPT
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_answer)
    )

    return RunnableParallel(context=retriever, question=RunnablePassthrough()).assign(answer=answer)


def get_chain() -> Runnable:
    """Cadena sobre la colección persistente de data/chroma/ y el modelo local."""
    return build_chain(default_retriever(get_vectorstore()), get_llm())


# =============================================================================
QUESTIONS = [
    "What drove JPMorgan Chase's net interest income in 2024?",
    'How did Bank of America deposits change in 2023?',
    'What risks does Goldman Sachs see from its Asset & Wealth Management business in 2025?',
    "What was Citigroup's CET1 capital ratio in 2024?",
    # Fuera de los reportes: debe responder "I do not know".
    "What was Apple's iPhone revenue in 2024?",
]


# =============================================================================
# Formas en que llama3.2:3b dice que el contexto no tiene la respuesta en vez
# de decir "I do not know" (y a veces sigue con algo que no se preguntó).
NO_ANSWER_PREFIXES = (
    'the context does not',
    'the provided context does not',
    'there is no information',
)


def is_no_answer(answer: str) -> bool:
    """True si el modelo dijo que el contexto no tiene la respuesta."""
    text = answer.strip().lower()
    return text.rstrip('.') == NO_ANSWER.lower() or text.startswith(NO_ANSWER_PREFIXES)


def clean_answer(answer: str) -> str:
    """Quita espacios y deja "I do not know." como exactamente `NO_ANSWER`."""
    return NO_ANSWER if is_no_answer(answer) else answer.strip()


def citations(docs: list[Document]) -> list[str]:
    """Empresa, año y sección de cada fuente, sin repetir y en orden de recuperación."""
    return list(dict.fromkeys(source_label(doc) for doc in docs))


def format_answer(result: dict) -> str:
    """Respuesta seguida de la empresa, el año y la sección de cada fuente."""
    lines = [f"Q: {result['question']}", f"A: {result['answer'].strip()}"]

    if is_no_answer(result['answer']):
        lines.append('Sources: none (the answer is not in the context)')
    else:
        lines.append('Sources:')
        lines += [f'  - {label}' for label in citations(result['context'])] or ['  - none retrieved']

    return '\n'.join(lines)


# =============================================================================
def main() -> None:
    """Hace las preguntas de prueba e imprime cada respuesta con sus fuentes.

    Raises:
        RuntimeError: Si la colección de Chroma está vacía.
    """
    store = get_vectorstore()
    if store._collection.count() == 0:
        raise RuntimeError('La colección está vacía; ejecuta `index_chunks` antes')

    chain = build_chain(default_retriever(store), get_llm())
    for n, question in enumerate(QUESTIONS, 1):
        print(f'\n[{n}] ' + format_answer(chain.invoke(question)))


# =============================================================================
if __name__ == '__main__':
    main()
