"""Pruebas del prompt text-to-SQL (sin el modelo)."""
import csv

import pytest

from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT
from financial_analyst_agent.sql import db, prompt

QUESTIONS_CSV = PROJECT_ROOT / 'eval' / 'questions.csv'


def render(question='How big is it?'):
    return prompt.build_sql_prompt().format_messages(question=question)


def test_system_message_has_schema_and_exact_metric_names():
    system = render()[0].content

    assert prompt.load_schema() in system
    for name in prompt.load_metric_names():
        assert f'- {name}:' in system
    assert '{' not in system  # no quedan variables sin rellenar


def test_examples_come_before_the_question():
    messages = render('Which bank is largest?')

    assert len(messages) == 1 + 2 * len(prompt.EXAMPLES) + 1
    assert messages[1].content == prompt.EXAMPLES[0][0]
    assert messages[2].content == prompt.EXAMPLES[0][1]
    assert messages[-1].content == 'Which bank is largest?'


def test_examples_are_not_eval_questions():
    with QUESTIONS_CSV.open(encoding='utf-8') as f:
        eval_questions = {row['question'].strip().lower() for row in csv.DictReader(f)}

    assert not {q.strip().lower() for q, _ in prompt.EXAMPLES} & eval_questions


@pytest.mark.skipif(not db.DB_PATH.exists(), reason='requiere data/processed/finance.duckdb')
@pytest.mark.parametrize('sql', [sql for _, sql in prompt.EXAMPLES])
def test_example_sql_returns_rows(sql):
    with db.connect() as con:
        assert con.execute(sql).fetchall()
