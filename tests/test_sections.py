"""Pruebas de la localización de Risk Factors y MD&A (sin red ni archivos)."""
from financial_analyst_agent.ingest import sections as sec

STANDARD_10K = [
    'Table of Contents',
    'Item 1A. Risk Factors',             # 1: índice
    'Item 1B. Unresolved Staff Comments',
    'Item 7. MD&A',                      # 3: índice
    'Item 7A. Market Risk',
    'Item 7. Management’s Discussion and Analysis',  # 5: glosario
    'Item 1A. Risk Factors',             # 6: sección real
    'Rates may rise.',
    'Credit may worsen.',
    'Item 1B. Unresolved Staff Comments',
    'Item 7. Management’s Discussion and Analysis',  # 10: sección real
    'Net income rose.',
    'Item 7A. Quantitative and Qualitative Disclosures',
]


def test_extract_sections_skips_index_and_glossary():
    result = sec.extract_sections('\n'.join(STANDARD_10K), 'GS')

    assert result['risk_factors']['ranges'] == [(6, 9)]
    assert result['risk_factors']['text'] == (
        'Item 1A. Risk Factors\nRates may rise.\nCredit may worsen.\n'
    )
    assert result['mdna']['ranges'] == [(10, 12)]


def test_locate_span_pick_first_for_repeated_page_headers():
    lines = [
        'Management’s discussion and analysis',
        'Intro.',
        'Management’s discussion and analysis',
        'More.',
        'Management’s report on internal control over financial reporting',
    ]
    span = sec.BANK_RULES['JPM']['mdna'][0]

    assert sec.locate_span(lines, span) == (0, 4)


def test_long_lines_are_not_headings():
    lines = ['Item 1A. ' + 'x' * 200, 'Item 1B.']

    assert sec.extract_sections('\n'.join(lines), 'GS')['risk_factors'] is None


def test_section_with_several_spans_joins_them_and_needs_all():
    lines = [
        'MANAGEMENT’S DISCUSSION AND ANALYSIS',
        'Part one.',
        'CAPITAL RESOURCES',
        'MANAGING GLOBAL RISK',
        'Part two.',
        'SIGNIFICANT ACCOUNTING POLICIES AND',
    ]

    mdna = sec.extract_sections('\n'.join(lines), 'C')['mdna']
    assert mdna['ranges'] == [(0, 2), (3, 5)]
    assert mdna['text'] == (
        'MANAGEMENT’S DISCUSSION AND ANALYSIS\nPart one.\n'
        'MANAGING GLOBAL RISK\nPart two.\n'
    )

    assert sec.extract_sections('\n'.join(lines[:3]), 'C')['mdna'] is None


def test_citi_headings_are_case_sensitive():
    lines = ['Risk factors', 'Text.', 'Sustainability']

    assert sec.extract_sections('\n'.join(lines), 'C')['risk_factors'] is None


def test_parse_report_name_uses_period_year_not_filing_year(tmp_path):
    path = tmp_path / 'JPM_2026-02-13_jpm-20251231.txt'

    assert sec.parse_report_name(path) == {
        'ticker': 'JPM', 'filing_date': '2026-02-13', 'fiscal_year': 2025,
    }


def test_save_report_sections_writes_text_and_metadata(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(sec, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(sec, 'RAW_10K_DIR', tmp_path / 'data' / 'raw' / '10k')
    monkeypatch.setattr(sec, 'SECTIONS_DIR', tmp_path / 'sections')
    monkeypatch.setattr(sec, 'MIN_SECTION_CHARS', 20)
    text_file = tmp_path / 'GS_2026-02-25_gs-20251231.txt'
    text_file.write_text('\n'.join(STANDARD_10K[:10]))
    stale = tmp_path / 'sections' / 'GS_2025_mdna.txt'
    stale.parent.mkdir()
    stale.write_text('old')

    # Sin la sección MD&A real, solo queda la del índice (13 caracteres).
    rows = sec.save_report_sections(text_file, {'GS': 'Goldman Sachs'})

    assert [(r['section'], r['chars']) for r in rows] == [
        ('risk_factors', len('\n'.join(STANDARD_10K[6:9]) + '\n')), ('mdna', 0),
    ]
    meta = json.loads((tmp_path / 'sections' / 'GS_2025_risk_factors.json').read_text())
    assert meta['company'] == 'Goldman Sachs'
    assert meta['fiscal_year'] == 2025
    assert meta['section'] == 'risk_factors'
    assert meta['source_file'] == 'data/raw/10k/GS_2026-02-25_gs-20251231.htm'
    assert not stale.exists()


def test_build_summary_flags_incomplete_reports():
    rows = [
        {'ticker': 'GS', 'fiscal_year': 2025, 'section': 'risk_factors', 'chars': 10},
        {'ticker': 'GS', 'fiscal_year': 2025, 'section': 'mdna', 'chars': 0},
        {'ticker': 'MS', 'fiscal_year': 2025, 'section': 'risk_factors', 'chars': 5},
        {'ticker': 'MS', 'fiscal_year': 2025, 'section': 'mdna', 'chars': 7},
    ]

    summary = sec.build_summary(rows)

    assert summary.columns.tolist() == [
        'ticker', 'fiscal_year', 'risk_factors', 'mdna', 'complete',
    ]
    assert summary['complete'].tolist() == [False, True]
