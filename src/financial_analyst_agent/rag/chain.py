"""Cadena RAG con LCEL: retriever de Chroma -> prompt -> llama3.2:3b -> texto.

La cadena devuelve la respuesta junto con los chunks recuperados, para
poder citar después la empresa, el año y la sección de cada fuente.
"""
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.rag.prompt import RAG_PROMPT, format_docs
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
