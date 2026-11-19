"""Pruebas de la evaluación con RAGAS (sin modelos ni red: la cadena es falsa)."""
import json
import math

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from financial_analyst_agent.rag import ragas_eval as re_


def fake_chain(fail_on=()):
    def run(question):
        if question in fail_on:
            raise RuntimeError('boom')
        return {'answer': f'A: {question}', 'context': [Document('c1'), Document('c2')]}
    return RunnableLambda(run)


def questions(n=3):
    return pd.DataFrame({'id': range(1, n + 1), 'question': [f'q{i}' for i in range(1, n + 1)]})


def test_repo_has_25_text_questions():
    text = re_.load_text_questions()

    assert len(text) == 25
    assert set(text['type']) == {'text'}
    assert text['id'].is_monotonic_increasing


def test_select_questions_by_ids_and_limit():
    qs = questions(5)

    assert re_.select_questions(qs)['id'].tolist() == [1, 2, 3, 4, 5]
    assert re_.select_questions(qs, ids=[4, 2])['id'].tolist() == [2, 4]
    assert re_.select_questions(qs, limit=2)['id'].tolist() == [1, 2]


def test_select_questions_rejects_unknown_ids():
    with pytest.raises(ValueError, match=r'\[9\]'):
        re_.select_questions(questions(3), ids=[1, 9])


def test_failed_question_is_recorded_not_dropped():
    row = re_.answer_question(fake_chain(fail_on={'q2'}), 'q2')

    assert row == {'answer': '', 'contexts': [], 'error': 'RuntimeError: boom'}


def test_answer_all_saves_and_retries_only_failures(tmp_path):
    path = tmp_path / 'answers.jsonl'
    calls = []

    def factory(fail_on=()):
        calls.append(fail_on)
        return fake_chain(fail_on)

    first = re_.answer_all(lambda: factory({'q2'}), questions(), path)
    assert [row['error'] for row in first] == ['', 'RuntimeError: boom', '']

    second = re_.answer_all(lambda: factory(), questions(), path)
    assert [row['answer'] for row in second] == ['A: q1', 'A: q2', 'A: q3']
    assert len(path.read_text().splitlines()) == 3

    re_.answer_all(lambda: factory(), questions(), path)
    assert len(calls) == 2  # la tercera vez no había nada pendiente
    assert json.loads(path.read_text().splitlines()[0])['contexts'] == ['c1', 'c2']


def test_build_results_keeps_every_question_and_config():
    answers = [
        {'id': 1, 'question': 'q1', 'answer': 'a', 'contexts': ['c'] * 5, 'error': ''},
        {'id': 3, 'question': 'q3', 'answer': 'I do not know', 'contexts': ['c'] * 5, 'error': ''},
        {'id': 2, 'question': 'q2', 'answer': '', 'contexts': [], 'error': 'RuntimeError: boom'},
    ]
    scores = pd.DataFrame({'faithfulness': [0.5, 0.0, math.nan], 'answer_relevancy': [0.9, 0.0, math.nan]})

    results = re_.build_results(answers, scores, '2026-11-17')

    assert list(results.columns) == re_.RESULT_COLUMNS
    assert results['id'].tolist() == [1, 3, 2]
    assert results['n_contexts'].tolist() == [5, 5, 0]
    assert results['no_answer'].tolist() == [False, True, False]
    assert results.loc[0, 'judge_model'] == re_.JUDGE_MODEL
    assert results.loc[0, 'k'] == 5
    assert 'NaN en id=[2]' in re_.summary(results)
    assert 'I do not know: 1/3' in re_.summary(results)
