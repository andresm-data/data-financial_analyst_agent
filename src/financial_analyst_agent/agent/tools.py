"""Nodos de herramientas del agente: `retrieve_text` (RAG) y `query_sql` (SQL).

Cada nodo lee `state["question"]` y devuelve una actualización parcial del
estado. Reutilizan el código existente sin cambiarlo:

- `retrieve_text` usa el retriever por defecto de la cadena RAG
  (`rag.chain.default_retriever`: búsqueda híbrida + rerank, k=5) y solo
  recupera; no genera respuesta. La respuesta la escribe el nodo `answer`.
- `query_sql` usa `sql.tool.answer_with_sql` (validador, solo lectura y
  reintentos) y guarda por separado el SQL ejecutado, las filas y el error.

Si una herramienta falla, el mensaje va a `error` y el grafo sigue; el manejo
completo de errores llega el 30 de noviembre.

Uso rápido:
    poetry run python -m financial_analyst_agent.agent.tools text "pregunta"
    poetry run python -m financial_analyst_agent.agent.tools sql "pregunta"
"""
import json
import sys
from functools import cache

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from financial_analyst_agent.agent.state import AgentState, Source
from financial_analyst_agent.rag.chain import default_retriever
from financial_analyst_agent.rag.prompt import source_label
from financial_analyst_agent.rag.retrieval import K
from financial_analyst_agent.rag.vectorstore import get_vectorstore
from financial_analyst_agent.sql.tool import answer_with_sql


# =============================================================================
@cache
def text_retriever() -> Runnable:
    """Retriever de la cadena RAG (k=5), creado una vez: carga Chroma y el reranker."""
    return default_retriever(get_vectorstore())


def to_source(doc: Document) -> Source:
    """Empresa, año, sección y archivo de origen de un chunk, con su texto."""
    meta = doc.metadata
    section = meta.get('section_title') or meta.get('section')
    if meta.get('heading'):
        section = f"{section} > {meta['heading']}"
    return Source(
        company=meta.get('company'), fiscal_year=meta.get('fiscal_year'), section=section,
        source_file=meta.get('source_file'), label=source_label(doc), text=doc.page_content,
    )


def join_errors(*errors: str | None) -> str | None:
    """Une los errores de varios nodos (ruta `both`) sin perder ninguno."""
    found = [e for e in errors if e]
    return ' | '.join(found) if found else None


# =============================================================================
def retrieve_text(state: AgentState, retriever: Runnable | None = None) -> AgentState:
    """Nodo: recupera los 5 chunks más relevantes y los guarda en `sources`."""
    try:
        docs = (retriever or text_retriever()).invoke(state['question'])[:K]
    except Exception as exc:  # p. ej. Chroma no disponible
        return {'sources': [], 'error': join_errors(state.get('error'), f'RAG error: {type(exc).__name__}: {exc}')}
    return {'sources': [to_source(doc) for doc in docs]}


def query_sql(state: AgentState, tool=answer_with_sql) -> AgentState:
    """Nodo: genera y ejecuta el SQL; guarda `sql`, `rows` y `error`."""
    try:
        result = tool(state['question'])
    except Exception as exc:  # p. ej. Ollama o la base no disponibles
        return {'sql': None, 'rows': [],
                'error': join_errors(state.get('error'), f'SQL error: {type(exc).__name__}: {exc}')}

    error = f"SQL error: {result['error']}" if result['error'] else None
    return {'sql': result['sql'], 'rows': result['rows'], 'error': join_errors(state.get('error'), error)}


# =============================================================================
def main() -> None:
    kind, question = sys.argv[1], ' '.join(sys.argv[2:])
    node = retrieve_text if kind == 'text' else query_sql
    update = node({'question': question})
    for source in update.get('sources', []):
        source['text'] = source['text'][:120] + '...'
    print(json.dumps(update, indent=2, default=str, ensure_ascii=False))


if __name__ == '__main__':
    main()
