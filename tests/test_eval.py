"""Pruebas de la evaluación del RAG (sin modelo ni Chroma)."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from financial_analyst_agent.rag import eval as ev

CET1 = ev.EvalQuestion('Citi CET1 2024?', ['C_0054'], ['13.63'], ['5.8'])
APPLE = ev.EvalQuestion('Apple iPhone revenue?')


def test_repo_questions_load_and_mix_answerable_and_not():
    questions = ev.load_questions()

    assert len(questions) >= 10
    assert any(q.answerable for q in questions)
    assert any(not q.answerable for q in questions)
    assert all(q.expected_chunks for q in questions if q.answerable)


def test_first_hit_rank():
    assert ev.first_hit_rank(['a', 'b', 'c'], ['c', 'b']) == 2
    assert ev.first_hit_rank(['a'], ['z']) is None


def test_retrieval_summary():
    assert ev.retrieval_summary([1, 2, None, None]) == {
        'questions': 4, 'hit_at_k': 0.5, 'mrr': pytest.approx((1 + 0.5) / 4),
    }
    assert ev.retrieval_summary([]) == {'questions': 0, 'hit_at_k': 0.0, 'mrr': 0.0}


@pytest.mark.parametrize('question, answer, grade', [
    (CET1, 'The CET1 ratio was 13.63%.', 'correct'),
    (CET1, 'I do not know', 'missed'),
    (CET1, 'The CET1 ratio was 5.8%.', 'wrong'),
    (CET1, 'It was 13.63%, or 5.8% leverage.', 'wrong'),
    (APPLE, 'I do not know', 'correct'),
    (APPLE, 'iPhone revenue was $200 billion.', 'invented'),
])
def test_grade_answer(question, answer, grade):
    assert ev.grade_answer(question, answer) == grade


def test_evaluate_retrieval_skips_unanswerable():
    retriever = RunnableLambda(lambda q: [Document(page_content='x', metadata={'chunk_id': 'C_0054'})])

    report = ev.evaluate_retrieval(retriever, [CET1, APPLE])

    assert [row['rank'] for row in report['rows']] == [1]
    assert report['summary']['hit_at_k'] == 1.0


def test_evaluate_answers_counts_grades():
    answers = iter(['13.63%', 'Apple sold many iPhones'])
    chain = RunnableLambda(lambda q: {'answer': next(answers)})

    report = ev.evaluate_answers(chain, [CET1, APPLE])

    assert report['summary'] == {'correct': 1, 'missed': 0, 'wrong': 0, 'invented': 1, 'questions': 2}


def test_save_report(tmp_path):
    path = ev.save_report({'a': 1}, tmp_path / 'eval')

    assert json.loads(path.read_text()) == {'a': 1}
