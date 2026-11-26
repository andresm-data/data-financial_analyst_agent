"""Pruebas del clasificador sin modelo de lenguaje (con un clasificador falso)."""
import csv
import difflib
import re
from pathlib import Path

import pytest
from langgraph.graph import END, START, StateGraph

from financial_analyst_agent.agent import classifier as clf
from financial_analyst_agent.agent.schemas import ROUTES, RouteDecision
from financial_analyst_agent.sql.prompt import load_metric_names

QUESTIONS_CSV = Path(__file__).resolve().parents[1] / 'eval' / 'questions.csv'
MAX_SIMILARITY = 0.8


def normalize(text: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split())


def eval_questions() -> list[str]:
    with QUESTIONS_CSV.open(encoding='utf-8') as f:
        return [row['question'] for row in csv.DictReader(f)]


def max_similarity(example: str, questions: list[str]) -> tuple[float, str]:
    return max((difflib.SequenceMatcher(None, normalize(example), normalize(q)).ratio(), q)
               for q in questions)


class FakeClassifier:
    def __init__(self, output):
        self.output = output

    def invoke(self, messages):
        return self.output


def test_two_examples_per_route():
    routes = [route for _, route, _ in clf.EXAMPLES]
    assert sorted(routes) == sorted(ROUTES * 2)


def test_examples_are_not_in_the_eval_set():
    questions = eval_questions()
    normalized = {normalize(q) for q in questions}
    for example, _, _ in clf.EXAMPLES:
        assert normalize(example) not in normalized
        ratio, closest = max_similarity(example, questions)
        assert ratio <= MAX_SIMILARITY, (example, closest, ratio)


def test_prompt_lists_every_metric_from_the_yaml_file():
    prompt = clf.build_system_prompt()
    for name, words in load_metric_names().items():
        assert f"- {name}: {', '.join(words)}" in prompt
    assert all(route in prompt for route in ROUTES)


def test_messages_end_with_the_user_question():
    messages = clf.build_messages('Why did deposits fall?')
    assert messages[-1].content == 'Question: Why did deposits fall?'


def test_classify_question_rejects_unparseable_output():
    with pytest.raises(ValueError):
        clf.classify_question('Hi?', classifier=FakeClassifier(None))


def test_classify_node_returns_route_and_reason_inside_a_graph(monkeypatch):
    decision = RouteDecision(route='numbers', reason='Asks for a value.')
    monkeypatch.setattr(clf, 'get_classifier', lambda: FakeClassifier(decision))

    builder = StateGraph(dict)
    builder.add_node('classify', clf.classify)
    builder.add_edge(START, 'classify')
    builder.add_edge('classify', END)
    out = builder.compile().invoke({'question': 'What were BAC deposits in 2024?'})

    assert out['route'] == 'numbers'
    assert out['route_reason'] == 'Asks for a value.'
