"""Pruebas del grafo del agente con nodos falsos (sin Ollama, Chroma ni DuckDB)."""
import pytest

from financial_analyst_agent.agent import graph as g
from financial_analyst_agent.agent.schemas import ROUTES


def fake_graph(route: str):
    return g.make_graph(
        classify_node=lambda s: {'route': route, 'route_reason': 'fake'},
        retrieve_node=lambda s: {'sources': []},
        sql_node=lambda s: {'sql': 'SELECT 1', 'rows': [], 'error': None},
    )


def node_order(graph, question: str = 'q') -> list[str]:
    return [node for update in graph.stream({'question': question}, stream_mode='updates')
            for node in update]


def test_every_route_has_a_target():
    assert set(g.ROUTE_TARGETS) == set(ROUTES)
    g.check_routes()
    with pytest.raises(ValueError):
        g.check_routes({r: 'refuse' for r in ROUTES[:-1]})


@pytest.mark.parametrize('route, expected', [
    ('text', ['classify', 'retrieve_text']),
    ('numbers', ['classify', 'query_sql']),
    ('both', ['classify', 'retrieve_text', 'query_sql']),
    ('out_of_scope', ['classify', 'refuse']),
])
def test_route_visits_expected_nodes(route, expected):
    assert node_order(fake_graph(route))[:len(expected)] == expected


def test_refuse_is_fixed_message():
    state = fake_graph('out_of_scope').invoke({'question': 'Should I buy?'})
    assert state['answer'] == g.REFUSE_MESSAGE


def test_build_graph_compiles_once():
    assert g.build_graph() is g.build_graph()
    assert {'classify', 'retrieve_text', 'query_sql', 'refuse'} <= set(g.build_graph().get_graph().nodes)
