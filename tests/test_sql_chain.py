"""Pruebas de la cadena text-to-SQL con un modelo falso (sin Ollama)."""
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_analyst_agent.sql.chain import build_sql_chain, clean_sql

SQL = "SELECT value FROM financials WHERE company = 'GS'"


@pytest.mark.parametrize('raw', [
    SQL,
    f'  \n{SQL}\n\n',
    f'```sql\n{SQL}\n```',
    f'```SQL\n{SQL}\n```',
    f'```\n{SQL}\n```',
    f'Here is the query:\n```sql\n{SQL}\n```\nIt returns the value.',
])
def test_clean_sql_removes_fences_and_whitespace(raw):
    assert clean_sql(raw) == SQL


def test_clean_sql_keeps_multiline_query():
    sql = 'SELECT company\nFROM financials\nLIMIT 1'
    assert clean_sql(f'```sql\n{sql}\n```') == sql


def test_chain_returns_clean_sql_from_model_output():
    llm = FakeListChatModel(responses=[f'```sql\n{SQL};\n```'])

    assert build_sql_chain(llm).invoke({'question': 'GS value?'}) == f'{SQL};'
