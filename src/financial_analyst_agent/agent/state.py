"""Estado compartido del grafo del agente (diseño: docs/agent_graph.md).

`total=False`: cada nodo devuelve solo los campos que escribe, y LangGraph los
une al estado. Los campos son los de la tabla "Planned state" del diseño.
"""
from typing import Any, TypedDict


class Source(TypedDict):
    """Un fragmento del 10-K recuperado por `retrieve_text`."""

    company: str
    fiscal_year: int
    section: str        # sección, con el apartado si lo hay ("MD&A > NET INTEREST INCOME")
    source_file: str    # archivo de la sección de origen (data/processed/sections/...)
    label: str          # "Empresa | FYaaaa | Sección", como en la cadena RAG
    text: str


class AgentState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    sources: list[Source]
    sql: str | None
    rows: list[dict[str, Any]]
    answer: str
    error: str | None
