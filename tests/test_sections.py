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
