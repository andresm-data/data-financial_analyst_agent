"""Esquema de la ruta que devuelve el clasificador del agente.

`ROUTES` es la lista compartida de rutas: el modelo Pydantic la usa para
validar y el grafo la reutilizará en `add_conditional_edges`. Debe coincidir
con las rutas de `docs/agent_graph.md`.
"""
from typing import Literal, get_args

from pydantic import BaseModel, Field

Route = Literal['text', 'numbers', 'both', 'out_of_scope']
ROUTES: tuple[str, ...] = get_args(Route)


class RouteDecision(BaseModel):
    """Decisión del clasificador: qué tipo de pregunta es y por qué."""

    route: Route = Field(
        description=(
            "Type of question. 'text': answered from the narrative of the 10-K "
            "(risks, causes, explanations). 'numbers': answered from the table of "
            "figures (values, rankings, growth). 'both': needs a figure and also "
            "an explanation from the report. 'out_of_scope': investment advice, "
            "price predictions, destructive or secret-seeking requests, or topics "
            "unrelated to the reports."
        ),
    )
    reason: str = Field(
        description='One short sentence explaining why this route was chosen.',
    )
