"""Pruebas del manifiesto de descargas (sin red)."""
import os

import pandas as pd
import pytest

from financial_analyst_agent.ingest import manifest as mf


def make_file(path, mtime=1_700_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    os.utime(path, (mtime, mtime))
    return path


def test_build_manifest_uses_relative_path_and_iso_utc_date(tmp_path):
    path = make_file(tmp_path / 'data' / 'raw' / 'a.json')

    df = mf.build_manifest([(path, 'https://sec.gov/a')], root=tmp_path)

    assert df.columns.tolist() == mf.COLUMNS
    assert df.iloc[0].tolist() == [
        'data/raw/a.json', 'https://sec.gov/a', '2023-11-14T22:13:20+00:00'
    ]


def test_build_manifest_keeps_previous_date_for_same_file_and_url(tmp_path):
    path = make_file(tmp_path / 'a.json')
    previous = {('a.json', 'https://sec.gov/a'): '2020-01-01T00:00:00+00:00'}

    df = mf.build_manifest(
        [(path, 'https://sec.gov/a')], previous, root=tmp_path
    )

    assert df['downloaded_at'].tolist() == ['2020-01-01T00:00:00+00:00']


def test_build_manifest_fails_if_a_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        mf.build_manifest([(tmp_path / 'nope.json', 'u')], root=tmp_path)


def test_downloaded_sources_lists_10k_and_facts():
    filings = pd.DataFrame([{
        'ticker': 'JPM', 'filingDate': '2026-02-13',
        'primaryDocument': 'jpm-20251231.htm', 'url': 'https://sec.gov/jpm',
    }])
    banks = [{'ticker': 'JPM', 'cik': '19617'}]

    sources = mf.downloaded_sources(filings, banks)

    assert [p.name for p, _ in sources] == [
        'JPM_2026-02-13_jpm-20251231.htm', 'JPM.json'
    ]
    assert sources[1][1].endswith('/CIK0000019617.json')


def test_write_manifest_round_trip_preserves_dates(tmp_path, monkeypatch):
    path = make_file(tmp_path / 'f.json')
    pd.DataFrame(columns=['ticker', 'filingDate', 'primaryDocument', 'url']) \
      .to_csv(tmp_path / 'filings.csv', index=False)
    monkeypatch.setattr(mf, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(mf, 'FILINGS_CSV', tmp_path / 'filings.csv')
    monkeypatch.setattr(mf, 'load_banks', lambda: [])
    monkeypatch.setattr(mf, 'downloaded_sources', lambda f, b: [(path, 'u')])
    out = tmp_path / 'manifest.csv'

    first = mf.write_manifest(out)
    os.utime(path, (1_800_000_000, 1_800_000_000))
    second = mf.write_manifest(out)

    assert first['downloaded_at'].tolist() == ['2023-11-14T22:13:20+00:00']
    assert second['downloaded_at'].tolist() == first['downloaded_at'].tolist()
