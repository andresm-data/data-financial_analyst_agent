"""Prueba de punta a punta del grafo del agente con 3 preguntas inventadas.

Las preguntas NO vienen de `eval/questions.csv` ni se añaden ahí: el
enrutamiento se mide el 1 Dic con las 40 preguntas y la prueba completa es el
7 Dic. Aparte, sin contarla, se comprueba que una pregunta de consejo de
inversión llega al nodo `refuse` provisional.

Uso: `poetry run python scripts/try_agent_graph.py`
"""
import csv
import json
import logging
import time
from pathlib import Path

from financial_analyst_agent.agent.graph import build_graph

QUESTIONS_CSV = Path(__file__).resolve().parents[1] / 'eval' / 'questions.csv'

CASES = [
    ('text', 'What does Goldman Sachs say about climate change risks in its 2024 10-K?'),
    ('numbers', "What were Morgan Stanley's total deposits at the end of fiscal year 2022?"),
    ('both', "How much net income did JPMorgan Chase report for fiscal year 2024, "
             "and what does its 2024 10-K say drove that result?"),
]
ADVICE = ('advice (not counted)', 'Should I buy Citigroup stock before its next earnings report?')


class UsageCollector(logging.Handler):
    """Guarda los tokens del prompt que registra el nodo `answer`."""

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record):
        if hasattr(record, 'answer_usage'):
            self.records.append(record.answer_usage)


def run(kind: str, question: str, usage: UsageCollector) -> dict:
    graph, state, order = build_graph(), {'question': question}, []
    usage.records.clear()
    start = time.perf_counter()
    for update in graph.stream(state, stream_mode='updates'):
        for node, values in update.items():
            order.append(node)
            state.update(values or {})
    seconds = time.perf_counter() - start

    print(f'\n=== [{kind}] {question}')
    print(f"Route: {state.get('route')} ({state.get('route_reason')})")
    print(f"Nodes: {' -> '.join(order)}")
    print('Sources:' if state.get('sources') else 'Sources: none')
    for s in state.get('sources', []):
        print(f"  - {s['company']} | FY{s['fiscal_year']} | {s['section']}")
    print(f"SQL: {state.get('sql')}")
    print(f"Rows: {json.dumps(state.get('rows'), default=str)}")
    print(f"Error: {state.get('error')}")
    for u in usage.records:
        print(f"Prompt tokens: {u['prompt_tokens']} / num_ctx {u['num_ctx']} ({u['prompt_chars']} chars)")
    print(f"Answer: {state.get('answer')}")
    print(f'Time: {seconds:.1f} s')
    return {'kind': kind, 'route': state.get('route'), 'nodes': order, 'seconds': seconds,
            'tokens': usage.records[0]['prompt_tokens'] if usage.records else None}


def main() -> None:
    with QUESTIONS_CSV.open(encoding='utf-8') as f:
        eval_questions = {row['question'].strip().lower() for row in csv.DictReader(f)}
    assert not any(q.lower() in eval_questions for _, q in [*CASES, ADVICE]), 'test question is in eval set'

    usage = UsageCollector()
    logging.getLogger('financial_analyst_agent.agent.answer').addHandler(usage)
    logging.getLogger('financial_analyst_agent.agent.answer').setLevel(logging.INFO)

    build_graph()  # compilar fuera del tiempo medido
    results = [run(kind, q, usage) for kind, q in CASES]
    advice = run(*ADVICE, usage)

    print('\n| Expected | Route | Nodes | Prompt tokens | Time (s) |\n|---|---|---|---|---|')
    for r in [*results, advice]:
        print(f"| {r['kind']} | {r['route']} | {' -> '.join(r['nodes'])} | {r['tokens'] or '-'} | {r['seconds']:.1f} |")


if __name__ == '__main__':
    main()
