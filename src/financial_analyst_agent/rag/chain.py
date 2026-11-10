"""Cadena RAG con LCEL: retriever de Chroma -> prompt -> llama3.2:3b -> texto.

La cadena devuelve la respuesta junto con los chunks recuperados, para
poder citar después la empresa, el año y la sección de cada fuente.
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.rag.prompt import NO_ANSWER, RAG_PROMPT, format_docs, source_label
from financial_analyst_agent.rag.retrieval import K, search
from financial_analyst_agent.rag.vectorstore import get_vectorstore


# =============================================================================
def get_retriever(store: Chroma, k: int = K) -> Runnable:
    """Retriever de Chroma con `k` resultados, filtrado por banco y año.

    Usa `retrieval.search`, que filtra por el banco y el año que nombra la
    pregunta antes de comparar vectores; sin ese filtro, los 10-K de otros
    bancos o años desplazan a los correctos.
    """
    return RunnableLambda(lambda question: search(store, question, k=k), name='chroma_retriever')


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
    )

    return RunnableParallel(context=retriever, question=RunnablePassthrough()).assign(answer=answer)


def get_chain() -> Runnable:
    """Cadena sobre la colección persistente de data/chroma/ y el modelo local."""
    return build_chain(get_retriever(get_vectorstore()), get_llm())


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
def is_no_answer(answer: str) -> bool:
    """True si el modelo dijo que el contexto no tiene la respuesta."""
    return answer.strip().rstrip('.').lower() == NO_ANSWER.lower()


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

    chain = build_chain(get_retriever(store), get_llm())
    for n, question in enumerate(QUESTIONS, 1):
        print(f'\n[{n}] ' + format_answer(chain.invoke(question)))


# =============================================================================
if __name__ == '__main__':
    main()
