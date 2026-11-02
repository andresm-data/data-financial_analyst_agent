"""Pruebas de load_financials con company facts sintéticos (sin red)."""

from financial_analyst_agent.ingest import load_financials as lf


def fact(val, end, filed, start=None, form='10-K', fp='FY', accn='a'):
    """Arma una fila de company facts como la devuelve la SEC."""
    row = {'val': val, 'end': end, 'filed': filed, 'form': form, 'fp': fp,
           'accn': accn, 'fy': int(filed[:4])}
    if start:
        row['start'] = start
    return row


def facts_with(tags: dict) -> dict:
    """Envuelve {etiqueta: [filas]} en la estructura facts > us-gaap."""
    return {'facts': {'us-gaap': {
        tag: {'units': {'USD': rows}} for tag, rows in tags.items()
    }}}


def test_annual_facts_uses_end_year_and_latest_filing():
    facts = facts_with({'NetIncomeLoss': [
        # 10-K de 2024 con el año 2024 y el comparativo de 2023
        fact(10, '2024-12-31', '2025-02-01', start='2024-01-01'),
        fact(8, '2023-12-31', '2025-02-01', start='2023-01-01'),
        # El 10-K de 2025 reexpresa 2024
        fact(11, '2024-12-31', '2026-02-01', start='2024-01-01'),
        # Trimestre dentro del 10-K: se descarta
        fact(3, '2024-12-31', '2025-02-01', start='2024-10-01'),
        # 10-Q: se descarta
        fact(99, '2024-06-30', '2024-08-01', start='2024-01-01',
             form='10-Q', fp='Q2'),
    ]})

    df = lf.annual_facts(facts, 'NetIncomeLoss')

    assert df['fiscal_year'].tolist() == [2023, 2024]
    assert df['value'].tolist() == [8, 11]
    assert str(df['filed_date'].iloc[1]) == '2026-02-01'


def test_annual_facts_instant_keeps_only_fiscal_year_end():
    facts = facts_with({'Assets': [
        fact(100, '2024-12-31', '2025-02-01'),
        fact(90, '2023-12-31', '2025-02-01'),
        # Saldo de apertura por cambio contable: se descarta
        fact(95, '2024-01-01', '2025-02-01'),
    ]})

    df = lf.annual_facts(facts, 'Assets')

    assert df['fiscal_year'].tolist() == [2023, 2024]
    assert df['value'].tolist() == [90, 100]


def test_annual_facts_missing_tag_returns_empty():
    assert lf.annual_facts(facts_with({}), 'Revenues').empty


def test_extract_metric_fills_missing_years_with_alternative():
    facts = facts_with({
        'Revenues': [fact(5, '2014-12-31', '2015-02-01', start='2014-01-01')],
        'RevenuesNetOfInterestExpense': [
            fact(6, '2015-12-31', '2016-02-01', start='2015-01-01'),
            fact(4, '2014-12-31', '2016-02-01', start='2014-01-01'),
        ],
    })

    df, used = lf.extract_metric(
        facts, 'Revenues', lf.METRIC_TAGS['Revenues']
    )

    assert df['value'].tolist() == [5, 6]
    assert used == {'Revenues': [2014], 'RevenuesNetOfInterestExpense': [2015]}


def test_extract_financials_columns_and_company():
    facts = facts_with({
        'Assets': [fact(100, '2024-12-31', '2025-02-01')],
        'ProfitLoss': [fact(7, '2024-12-31', '2025-02-01', start='2024-01-01')],
    })

    df = lf.extract_financials(facts, 'TFC')

    assert list(df.columns) == lf.COLUMNS
    assert set(df['metric']) == {'Assets', 'NetIncomeLoss'}
    assert (df['company'] == 'TFC').all()


def test_load_to_duckdb_is_repeatable(tmp_path):
    facts = facts_with({
        'Assets': [
            fact(100, '2024-12-31', '2025-02-01'),
            fact(90, '2023-12-31', '2025-02-01'),
        ],
    })
    table = lf.extract_financials(facts, 'JPM')
    db_path = tmp_path / 'finance.duckdb'

    lf.load_to_duckdb(table, db_path)
    lf.load_to_duckdb(table, db_path)

    counts = lf.verify(db_path)
    assert counts.iloc[:, 1].tolist() == [2]
