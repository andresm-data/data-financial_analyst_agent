"""Pruebas del modelo de ruta del clasificador (sin modelo de lenguaje)."""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_analyst_agent.agent.schemas import ROUTES, RouteDecision

AGENT_GRAPH_DOC = Path(__file__).resolve().parents[1] / 'docs' / 'agent_graph.md'


def test_routes_match_the_design_doc():
    text = AGENT_GRAPH_DOC.read_text(encoding='utf-8')
    line = next(l for l in text.splitlines() if l.startswith('Routes:'))
    assert tuple(re.findall(r'`([a-z_]+)`', line)) == ROUTES


@pytest.mark.parametrize('route', ROUTES)
def test_accepts_every_valid_route(route):
    decision = RouteDecision(route=route, reason='Short reason.')
    assert decision.route == route


def test_rejects_an_invalid_route():
    with pytest.raises(ValidationError):
        RouteDecision(route='sql', reason='Not a route.')


def test_requires_a_reason():
    with pytest.raises(ValidationError):
        RouteDecision(route='text')
