"""Pruebas de la evaluación de tablas contra el HTML (sin datos descargados)."""
from financial_analyst_agent.rag import table_eval as te

HTML = """
<html><body>
<p>The table below presents net income.</p>
<table>
  <tr><td>(Dollars in millions)</td><td>2024</td><td>2023</td></tr>
  <tr><td>Net income</td><td>$ 27,132</td><td>$ 26,515</td></tr>
  <tr><td>Revenue</td><td>101,887</td><td>98,581</td></tr>
</table>
<table>
  <tr><td>•</td><td>Rates may rise.</td></tr>
  <tr><td>•</td><td>Credit may worsen.</td></tr>
</table>
</body></html>
"""

TEXT = '\n'.join([
    'Net income grew. The table below presents net income.',
    '(Dollars in millions) 2024 2023',
    'Net income $ 27,132 $ 26,515',
    'Revenue 101,887 98,581',
    'Rates may rise.',
])


def test_data_table_lines_skip_layout_tables():
    assert te.data_table_lines(HTML) == {
        '(Dollars in millions) 2024 2023',
        'Net income $ 27,132 $ 26,515',
        'Revenue 101,887 98,581',
    }


def test_predicted_table_lines_exclude_intro_sentence():
    lines, predicted = te.predicted_table_lines(TEXT)

    assert lines[1] == 'The table below presents net income.'
    assert [lines[i] for i in sorted(predicted)] == TEXT.splitlines()[1:4]
