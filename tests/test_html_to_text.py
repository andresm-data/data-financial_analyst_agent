"""Pruebas de la conversión de 10-K HTML a texto (sin red)."""
import os

import pytest

from financial_analyst_agent.ingest import html_to_text as ht


def test_html_to_text_drops_script_style_and_xbrl_header():
    html = (
        '<html><head><style>p {color: red}</style></head><body>'
        '<div style="display:none"><ix:header>hidden 123</ix:header></div>'
        '<script>alert(1)</script><p>Item 1A.&#160;Risk Factors</p>'
        '</body></html>'
    )

    assert ht.html_to_text(html) == 'Item 1A. Risk Factors\n'


def test_html_to_text_keeps_inline_spans_together_and_splits_blocks():
    html = (
        '<div><span>Manage</span><span>ment’s Discussion</span></div>'
        '<table><tr><td>Net income</td><td>$ 10</td></tr></table>'
    )

    assert ht.html_to_text(html) == 'Management’s Discussion\nNet income $ 10\n'


def test_convert_file_writes_txt_and_skips_if_up_to_date(tmp_path, monkeypatch):
    monkeypatch.setattr(ht, 'TEXT_DIR', tmp_path / 'text')
    raw = tmp_path / 'JPM_2026-02-13_jpm-20251231.htm'
    raw.write_text('<p>Hello</p>')
    os.utime(raw, (1_700_000_000, 1_700_000_000))

    target = ht.convert_file(raw)
    assert target == tmp_path / 'text' / 'JPM_2026-02-13_jpm-20251231.txt'
    assert target.read_text() == 'Hello\n'

    target.write_text('cached')
    ht.convert_file(raw)
    assert target.read_text() == 'cached'


def test_main_fails_without_raw_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ht, 'RAW_10K_DIR', tmp_path)

    with pytest.raises(FileNotFoundError):
        ht.main()
