"""Grafo LangGraph del agente (diseño: docs/agent_graph.md).

    START -> classify -> (ruta) -> retrieve_text / query_sql / refuse
    text:         retrieve_text -> answer -> END
    numbers:      query_sql -> answer -> END
    both:         retrieve_text -> query_sql -> answer -> END   (en secuencia)
    out_of_scope: refuse -> END

`build_graph()` compila el grafo una sola vez. `make_graph()` deja cambiar los
nodos por versiones falsas en las pruebas.
"""
from functools import cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from financial_analyst_agent.agent.answer import answer
from financial_analyst_agent.agent.classifier import classify
from financial_analyst_agent.agent.schemas import ROUTES
from financial_analyst_agent.agent.state import AgentState
from financial_analyst_agent.agent.tools import query_sql, retrieve_text

# Destino de cada ruta del clasificador. Debe cubrir exactamente `ROUTES`.
ROUTE_TARGETS: dict[str, str] = {
    'text': 'retrieve_text',
    'numbers': 'query_sql',
    'both': 'retrieve_text',
    'out_of_scope': 'refuse',
}

# PROVISIONAL hasta el 30 de noviembre: mensaje fijo. El nodo `refuse`
# definitivo (con el motivo del rechazo) se construye ese día.
REFUSE_MESSAGE = (
    'I can only answer questions about the banks\' 10-K annual reports and their yearly '
    'figures. I cannot give investment advice, predict prices or run that request.'
)


# =============================================================================
def refuse(state: AgentState) -> AgentState:
    """Nodo PROVISIONAL (hasta el 30 Nov): devuelve siempre el mismo mensaje corto."""
    return {'answer': REFUSE_MESSAGE}


def route_after_classify(state: AgentState) -> str:
    return state['route']


# Salidas de `retrieve_text`, con el nombre de la ruta como clave para que el
# dibujo del grafo muestre las mismas etiquetas que el diseño.
RETRIEVE_TARGETS: dict[str, str] = {'both': 'query_sql', 'text': 'answer'}


def route_after_retrieve(state: AgentState) -> str:
    """Ruta `both`: después del texto van las cifras; si no, a `answer`."""
    return 'both' if state.get('route') == 'both' else 'text'


def check_routes(targets: dict[str, str] = ROUTE_TARGETS) -> None:
    """Falla si alguna ruta del clasificador no tiene destino (o sobra alguna)."""
    if set(targets) != set(ROUTES):
        raise ValueError(f'Route targets {sorted(targets)} do not match ROUTES {sorted(ROUTES)}')


# =============================================================================
def make_graph(
    classify_node=classify, retrieve_node=retrieve_text, sql_node=query_sql, refuse_node=refuse,
    answer_node=answer,
) -> CompiledStateGraph:
    """Construye y compila el grafo con los nodos dados."""
    check_routes()
    graph = StateGraph(AgentState)
    graph.add_node('classify', classify_node)
    graph.add_node('retrieve_text', retrieve_node)
    graph.add_node('query_sql', sql_node)
    graph.add_node('refuse', refuse_node)
    graph.add_node('answer', answer_node)

    graph.add_edge(START, 'classify')
    graph.add_conditional_edges('classify', route_after_classify, ROUTE_TARGETS)
    graph.add_conditional_edges('retrieve_text', route_after_retrieve, RETRIEVE_TARGETS)
    graph.add_edge('query_sql', 'answer')
    graph.add_edge('answer', END)
    graph.add_edge('refuse', END)
    return graph.compile()


@cache
def build_graph() -> CompiledStateGraph:
    """El grafo del agente, compilado una sola vez por proceso."""
    return make_graph()
