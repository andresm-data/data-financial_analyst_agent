"""Pruebas del chunking de las secciones (sin red ni datos descargados)."""
import json

import pytest

from financial_analyst_agent.rag import chunking as ch

TEXT = '\n'.join(f'Line {i} talks about credit risk and liquidity.' for i in range(200))


def write_config(tmp_path, size, overlap):
    path = tmp_path / 'rag.yaml'
    path.write_text(
        f'chunking:\n  chunk_size: {size}\n  chunk_overlap: {overlap}\n',
        encoding='utf-8',
    )
    return path


def test_repo_config_starts_with_1000_and_150():
    assert ch.load_chunking_config() == ch.ChunkingConfig(1000, 150)


def test_chunks_respect_size_from_config(tmp_path):
    config = ch.load_chunking_config(write_config(tmp_path, 300, 50))
    chunks = ch.chunk_text(TEXT, config)

    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_changing_config_changes_result(tmp_path):
    small = ch.chunk_text(TEXT, ch.load_chunking_config(write_config(tmp_path, 300, 0)))
    large = ch.chunk_text(TEXT, ch.load_chunking_config(write_config(tmp_path, 1000, 0)))

    assert len(small) > len(large)


def test_overlap_repeats_text_between_chunks():
    chunks = ch.chunk_text(TEXT, ch.ChunkingConfig(300, 100))

    assert chunks[1].splitlines()[0] in chunks[0].splitlines()


@pytest.mark.parametrize('size, overlap', [(0, 0), (100, 100), (100, -1)])
def test_invalid_config_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        ch.ChunkingConfig(size, overlap)


def write_section(tmp_path, ticker='GS', year=2024, section='mdna'):
    section_file = f'data/processed/sections/{ticker}_{year}_{section}.txt'
    (tmp_path / section_file).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / section_file).write_text(TEXT, encoding='utf-8')
    metadata_file = tmp_path / section_file.replace('.txt', '.json')
    metadata_file.write_text(json.dumps({
        'company': 'Goldman Sachs',
        'ticker': ticker,
        'fiscal_year': year,
        'filing_date': '2025-02-27',
        'section': section,
        'section_title': "Management's Discussion and Analysis",
        'source_file': 'data/raw/10k/GS_2025-02-27_gs-20241231.htm',
        'text_file': 'data/processed/text/GS_2025-02-27_gs-20241231.txt',
        'section_file': section_file,
        'line_ranges': [[0, 200]],
        'chars': len(TEXT),
    }), encoding='utf-8')
    return metadata_file


def test_every_chunk_has_section_metadata(tmp_path):
    chunks = ch.chunk_section(
        write_section(tmp_path), ch.ChunkingConfig(300, 50), root=tmp_path
    )

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata == {
            'company': 'Goldman Sachs',
            'ticker': 'GS',
            'fiscal_year': 2024,
            'filing_date': '2025-02-27',
            'section': 'mdna',
            'section_title': "Management's Discussion and Analysis",
            'source_file': 'data/raw/10k/GS_2025-02-27_gs-20241231.htm',
            'section_file': 'data/processed/sections/GS_2024_mdna.txt',
            'chunk_index': i,
            'chunk_id': f'GS_2024_mdna_{i:04d}',
        }


def test_save_and_load_chunks_round_trip(tmp_path):
    chunks = ch.chunk_section(
        write_section(tmp_path), ch.ChunkingConfig(300, 50), root=tmp_path
    )
    path = tmp_path / 'chunks.jsonl'
    ch.save_chunks(chunks, path)

    assert ch.load_chunks(path) == chunks


def test_chunk_counts_by_bank_and_section(tmp_path):
    config = ch.ChunkingConfig(300, 50)
    chunks = (
        ch.chunk_section(write_section(tmp_path, section='mdna'), config, root=tmp_path)
        + ch.chunk_section(write_section(tmp_path, section='risk_factors'), config, root=tmp_path)
    )
    counts = ch.chunk_counts(chunks).set_index('ticker')

    assert counts.loc['GS', 'mdna'] == counts.loc['GS', 'risk_factors'] > 0
    assert counts.loc['TOTAL', 'total'] == len(chunks)
