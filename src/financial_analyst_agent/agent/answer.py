"""Nodo `answer`: une los fragmentos del 10-K y las filas del SQL en una respuesta.

Hace una sola llamada al modelo con todo el contexto disponible. Las reglas
son las del prompt estricto de la cadena RAG (`rag.prompt.SYSTEM_RULES`, sin
cambiarlo) más cuatro reglas: citar las fuentes, las unidades, responder las dos partes y,
si la cifra está en la tabla y en el texto, usar la de la tabla (la que se
compara en la evaluación). El modelo de 3B no siempre cumple esta última regla
(copia la cifra redondeada del 10-K), así que cuando hay texto y filas a la vez
el código añade las cifras de la tabla como valor de referencia.

Sin contexto (ni fragmentos ni filas) no se llama al modelo: la respuesta es
un mensaje fijo que dice qué falló o que no se sabe, sin inventar datos. Si una
herramienta falló pero hay otra fuente, se responde con esa fuente y se añade
una nota con el error.

El modelo de 3B no siempre cita bien, así que al final de cada respuesta se
añade un bloque `Sources:` generado en código: empresa, año y sección de cada
fragmento recuperado y, si hubo filas, la tabla de cifras (en dólares).

Cada llamada registra en el log `financial_analyst_agent.agent.answer` los
tokens del prompt que contó Ollama y los compara con `NUM_CTX`: Ollama recorta
el prompt sin avisar si no cabe. Por eso al modelo solo llegan las primeras
`MAX_PROMPT_ROWS` filas del SQL (la herramienta puede devolver hasta 100); el
estado conserva todas y la respuesta dice cuántas se omitieron.
"""
import json
import logging
from functools import cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from financial_analyst_agent.agent.state import AgentState
from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.rag.chain import clean_answer, is_no_answer
from financial_analyst_agent.rag.prompt import NO_ANSWER, SYSTEM_RULES

logger = logging.getLogger(__name__)

# Ventana de contexto con la que Ollama 0.17 carga llama3.2:3b (`ollama ps`):
# `get_llm` no fija `num_ctx`, así que se usa el valor por defecto del servidor.
NUM_CTX = 4096
NEAR_LIMIT = 0.8  # avisar si el prompt pasa del 80 % de la ventana
# Filas del SQL que entran al prompt. ~30 tokens por fila: 20 filas + 5 chunks
# de 1000 caracteres + reglas quedan por debajo del 80 % de NUM_CTX.
MAX_PROMPT_ROWS = 20

EXTRA_RULES = """6. After each fact taken from a 10-K excerpt, cite the header of that excerpt, for example (Goldman Sachs | FY2024 | Risk Factors).
7. Figures from the SQL result are in US dollars; a column ending in _usd_bn is in billions of US dollars. Write them as "USD <value> billion" with the value copied exactly, without rounding, and cite them as (financials table), never as a 10-K section.
8. If the question asks for a figure and an explanation, answer both parts, each only from its own context.
9. If the SQL result and a 10-K excerpt both give the asked figure, report the value from the SQL result."""

ANSWER_RULES = f'{SYSTEM_RULES}\n{EXTRA_RULES}'

NO_CONTEXT = f'{NO_ANSWER}. No 10-K passage or figure matched the question.'
NO_ROWS = f'{NO_ANSWER}. The query on the financials table returned no rows.'


# =============================================================================
def format_sources(sources: list[dict]) -> str:
    """Igual que `rag.prompt.format_docs`: cada fragmento con su fuente como encabezado."""
    return '\n\n'.join(f"[{n}] {s['label']}\n{s['text']}" for n, s in enumerate(sources, 1))


def format_rows(sql: str | None, rows: list[dict], max_rows: int = MAX_PROMPT_ROWS) -> str:
    """El SQL ejecutado y una fila por línea (como mucho `max_rows`), con los valores de DuckDB."""
    lines = [f'SQL: {sql}', 'Rows:']
    lines += [json.dumps(row, default=str) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f'(only the first {max_rows} of {len(rows)} rows are shown)')
    return '\n'.join(lines)


# Filas que se citan como cifra de referencia cuando hay texto y tabla a la vez.
MAX_FIGURE_ROWS = 5


def table_figures(rows: list[dict], max_rows: int = MAX_FIGURE_ROWS) -> str:
    """Las filas del SQL tal cual, como cifra de referencia (la tabla manda sobre el texto)."""
    lines = [', '.join(f'{k} = {v}' for k, v in row.items()) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f'... and {len(rows) - max_rows} more rows')
    return ('Figures from the financials table (reference values; _usd_bn = billions of US dollars):\n'
            + '\n'.join(f'  - {line}' for line in lines))


def truncation_note(rows: list[dict] | None, max_rows: int = MAX_PROMPT_ROWS) -> str | None:
    """Aviso fijo si el modelo no vio todas las filas."""
    if rows and len(rows) > max_rows:
        return f'Note: the answer only used the first {max_rows} of {len(rows)} rows returned by the SQL query.'
    return None


def build_messages(state: AgentState) -> list:
    """Prompt de la única llamada: reglas + los contextos que haya + pregunta."""
    parts = []
    if state.get('sources'):
        parts.append(f"10-K excerpts:\n{format_sources(state['sources'])}")
    if state.get('rows'):
        parts.append(f"SQL result from the financials table:\n{format_rows(state.get('sql'), state['rows'])}")
    human = 'Context:\n' + '\n\n'.join(parts) + f"\n\nQuestion: {state['question']}"
    return [SystemMessage(ANSWER_RULES), HumanMessage(human)]


def no_context_answer(state: AgentState) -> str:
    """Respuesta fija cuando no hay nada que dar al modelo."""
    if state.get('error'):
        return f"{NO_ANSWER}. The data could not be retrieved: {state['error']}"
    if state.get('sql') is not None:
        return NO_ROWS
    return NO_CONTEXT


def sources_block(state: AgentState) -> str:
    """Fuentes del contexto sin repetir, en orden de recuperación (como `rag.chain.citations`)."""
    labels = list(dict.fromkeys(s['label'] for s in state.get('sources') or []))
    if state.get('rows'):
        labels.append('financials table (SQL query above; amounts in US dollars)')
    return 'Sources:\n' + '\n'.join(f'  - {label}' for label in labels)


def log_usage(response, prompt_chars: int) -> None:
    tokens = (response.usage_metadata or {}).get('input_tokens', 0)
    fields = {'prompt_tokens': tokens, 'prompt_chars': prompt_chars, 'num_ctx': NUM_CTX}
    near = tokens >= NEAR_LIMIT * NUM_CTX
    logger.log(logging.WARNING if near else logging.INFO,
               'answer prompt_tokens=%d num_ctx=%d (%.0f%%)%s', tokens, NUM_CTX, 100 * tokens / NUM_CTX,
               ' NEAR LIMIT: Ollama may truncate the prompt' if near else '', extra={'answer_usage': fields})


# =============================================================================
@cache
def answer_llm() -> BaseChatModel:
    return get_llm(temperature=0.0)


def answer(state: AgentState, llm: BaseChatModel | None = None) -> AgentState:
    """Nodo: escribe `answer` con una llamada al modelo, o un mensaje fijo sin contexto."""
    if not state.get('sources') and not state.get('rows'):
        return {'answer': no_context_answer(state)}

    messages = build_messages(state)
    response = (llm or answer_llm()).invoke(messages)
    log_usage(response, sum(len(m.content) for m in messages))

    text = clean_answer(response.content)
    if not is_no_answer(text):
        if state.get('sources') and state.get('rows'):
            text += f"\n\n{table_figures(state['rows'])}"
        text += f'\n\n{sources_block(state)}'
    if note := truncation_note(state.get('rows')):
        text += f'\n\n{note}'
    if state.get('error'):
        text += f"\n\nNote: part of the data could not be retrieved: {state['error']}"
    return {'answer': text}
