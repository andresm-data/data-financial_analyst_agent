"""Pruebas de la limpieza de pies y títulos de página."""
from financial_analyst_agent.rag import cleaning as cl

COMPANY = 'Bank of America'
TITLE = "Management's Discussion and Analysis"
PROSE = 'Net income rose because revenue grew and expenses were flat.'


def test_page_numbers_follow_sequence_and_table_cells_stay():
    lines = [PROSE, '26', PROSE, 'Other', '749', 'Total', '27', PROSE, '29']

    assert cl.page_number_lines(lines) == {1, 6, 8}


def test_page_numbers_in_two_separate_page_ranges():
    lines = ['8', PROSE, '9', PROSE, '64', PROSE, '65']

    assert cl.page_number_lines(lines) == {0, 2, 4, 6}


def test_company_footer_needs_company_name_and_edge_number():
    assert cl.is_company_footer('Bank of America 32', COMPANY)
    assert cl.is_company_footer('26 Bank of America', COMPANY)
    assert not cl.is_company_footer('Bank of America', COMPANY)
    assert not cl.is_company_footer('Table 38', COMPANY)
    assert not cl.is_company_footer('December 31', COMPANY)


def test_running_headers_only_title_company_or_contents():
    lines = (
        ['Management’s discussion and analysis'] * 10
        + ['Table of Contents'] * 10
        + ['As of December'] * 10
        + ['BANK OF AMERICA CORPORATION AND SUBSIDIARIES'] * 10
    )

    assert cl.running_headers(lines, COMPANY, TITLE) == {
        'Management’s discussion and analysis',
        'Table of Contents',
        'BANK OF AMERICA CORPORATION AND SUBSIDIARIES',
    }


def test_clean_section_text_removes_page_furniture():
    text = '\n'.join([
        PROSE, 'Bank of America 30', 'Bank of America Corp./2024 Form 10-K',
        'Off-Balance Sheet Arrangement Disclosure in Form 10-K', 'Table 38', PROSE,
    ])

    assert cl.clean_section_text(text, COMPANY, TITLE) == '\n'.join([
        PROSE, 'Off-Balance Sheet Arrangement Disclosure in Form 10-K', 'Table 38', PROSE,
    ]) + '\n'
