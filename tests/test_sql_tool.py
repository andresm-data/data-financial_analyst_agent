"""Pruebas de la herramienta text-to-SQL con una base temporal y un modelo falso."""
import csv
from functools import partial

import duckdb
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT
from financial_analyst_agent.sql import db, tool
from financial_analyst_agent.sql.chain import build_sql_chain


@pytest.fixture
def connect(tmp_path):
    path = tmp_path / 'tiny.duckdb'
    with duckdb.connect(str(path)) as con:
        con.execute('CREATE TABLE financials (company VARCHAR, value DOUBLE)')
        con.execute('INSERT INTO financials SELECT \'JPM\', i FROM range(150) t(i)')
    return partial(db.connect, path)


def test_run_sql_returns_rows_with_column_names(connect):
    result = tool.run_sql('SELECT company, value AS v FROM financials WHERE value < 2 ORDER BY v', connect)

    assert result['error'] is None
    assert result['columns'] == ['company', 'v']
    assert result['rows'] == [{'company': 'JPM', 'v': 0.0}, {'company': 'JPM', 'v': 1.0}]


def test_run_sql_caps_rows(connect):
    assert len(tool.run_sql('SELECT * FROM financials', connect)['rows']) == tool.MAX_ROWS


@pytest.mark.parametrize('sql, error', [
    ('SELECT nope FROM financials', 'BinderException'),
    ('SELEC 1', 'ParserException'),
    ('DELETE FROM financials', 'read-only'),
])
def test_run_sql_returns_error_instead_of_rows(connect, sql, error):
    result = tool.run_sql(sql, connect)

    assert error in result['error']
    assert result['rows'] == [] and result['sql'] == sql


def test_answer_with_sql_runs_generated_query(connect):
    chain = build_sql_chain(FakeListChatModel(responses=['```sql\nSELECT count(*) AS n FROM financials\n```']))

    result = tool.answer_with_sql('How many rows?', chain, connect)

    assert result['sql'] == 'SELECT\n  COUNT(*) AS n\nFROM financials\nLIMIT 100'
    assert result['rows'] == [{'n': 150}]


def test_tool_takes_a_question():
    assert tool.text_to_sql.name == 'text_to_sql'
    assert set(tool.text_to_sql.args) == {'question'}


def test_test_questions_are_not_eval_questions():
    with (PROJECT_ROOT / 'eval' / 'questions.csv').open(encoding='utf-8') as f:
        eval_questions = {row['question'].strip().lower() for row in csv.DictReader(f)}

    assert len(tool.TEST_QUESTIONS) == 5
    assert not {q.strip().lower() for q in tool.TEST_QUESTIONS} & eval_questions
