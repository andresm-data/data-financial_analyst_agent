"""Ejercicio de aprendizaje de LangGraph: un grafo de dos nodos sin modelo.

No es el agente. Sirve para ver las piezas básicas con la versión instalada:
un estado `TypedDict`, dos nodos deterministas, las aristas
START -> normalize -> write_answer -> END, `compile()` e `invoke()`.
Al final imprime el diagrama Mermaid con `get_graph().draw_mermaid()`.

Uso: `poetry run python scripts/langgraph_two_nodes.py`
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    answer: str


def normalize(state: State) -> dict:
    """Quita espacios sobrantes y asegura que la pregunta termine en '?'."""
    question = ' '.join(state['question'].split())
    if not question.endswith('?'):
        question += '?'
    return {'question': question}


def write_answer(state: State) -> dict:
    """Escribe una respuesta de ejemplo; solo devuelve la clave que cambia."""
    return {'answer': f'(demo) You asked: {state["question"]}'}


def build_graph():
    builder = StateGraph(State)
    builder.add_node('normalize', normalize)
    builder.add_node('write_answer', write_answer)
    builder.add_edge(START, 'normalize')
    builder.add_edge('normalize', 'write_answer')
    builder.add_edge('write_answer', END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    state_in: State = {'question': '   what was JPM net income   in 2024 ', 'answer': ''}
    state_out = graph.invoke(state_in)
    print('Input state: ', state_in)
    print('Output state:', state_out)
    print()
    print(graph.get_graph().draw_mermaid())


if __name__ == '__main__':
    main()
