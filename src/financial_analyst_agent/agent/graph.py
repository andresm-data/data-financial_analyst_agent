"""Grafo LangGraph del agente (diseño: docs/agent_graph.md).

    START -> classify -> (ruta) -> retrieve_text / query_sql / refuse
    text:         retrieve_text -> END
    numbers:      query_sql -> END
    both:         retrieve_text -> query_sql -> END   (en secuencia)
    out_of_scope: refuse -> END

`build_graph()` compila el grafo una sola vez. `make_graph()` deja cambiar los
nodos por versiones falsas en las pruebas.
"""
from functools import cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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


def route_after_retrieve(state: AgentState) -> str:
    """Ruta `both`: después del texto van las cifras; si no, se termina."""
    return 'query_sql' if state.get('route') == 'both' else 'next'


def check_routes(targets: dict[str, str] = ROUTE_TARGETS) -> None:
    """Falla si alguna ruta del clasificador no tiene destino (o sobra alguna)."""
    if set(targets) != set(ROUTES):
        raise ValueError(f'Route targets {sorted(targets)} do not match ROUTES {sorted(ROUTES)}')


# =============================================================================
def make_graph(
    classify_node=classify, retrieve_node=retrieve_text, sql_node=query_sql, refuse_node=refuse,
) -> CompiledStateGraph:
    """Construye y compila el grafo con los nodos dados."""
    check_routes()
    graph = StateGraph(AgentState)
    graph.add_node('classify', classify_node)
    graph.add_node('retrieve_text', retrieve_node)
    graph.add_node('query_sql', sql_node)
    graph.add_node('refuse', refuse_node)

    graph.add_edge(START, 'classify')
    graph.add_conditional_edges('classify', route_after_classify, ROUTE_TARGETS)
    # Temporal: hasta que exista `answer`, las herramientas terminan en END.
    graph.add_conditional_edges('retrieve_text', route_after_retrieve, {'query_sql': 'query_sql', 'next': END})
    graph.add_edge('query_sql', END)
    graph.add_edge('refuse', END)
    return graph.compile()


@cache
def build_graph() -> CompiledStateGraph:
    """El grafo del agente, compilado una sola vez por proceso."""
    return make_graph()
