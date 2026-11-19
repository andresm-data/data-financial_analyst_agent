"""Pruebas de la conexión de solo lectura a DuckDB (con una base temporal)."""
import duckdb
import pytest

from financial_analyst_agent.sql import db


@pytest.fixture
def tiny_db(tmp_path):
    path = tmp_path / 'tiny.duckdb'
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE financials (company VARCHAR, value DOUBLE)")
        con.execute("INSERT INTO financials VALUES ('JPM', 1.0)")
    return path


def test_connect_reads(tiny_db):
    with db.connect(tiny_db) as con:
        assert con.execute('SELECT company, value FROM financials').fetchall() == [('JPM', 1.0)]


@pytest.mark.parametrize('sql', [
    "INSERT INTO financials VALUES ('BAC', 2.0)",
    'DELETE FROM financials',
    'CREATE TABLE other (x INTEGER)',
])
def test_connect_rejects_writes(tiny_db, sql):
    with db.connect(tiny_db) as con, pytest.raises(duckdb.InvalidInputException, match='read-only'):
        con.execute(sql)


def test_connect_does_not_create_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        db.connect(tmp_path / 'missing.duckdb')
    assert not (tmp_path / 'missing.duckdb').exists()
