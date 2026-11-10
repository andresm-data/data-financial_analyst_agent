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
    assert ch.load_chunking_config() == ch.ChunkingConfig(
        1000, 150, ('\n\n', '\n', '. ', '; ', ' ', ''), min_chunk_chars=200
    )


def test_long_paragraph_is_cut_at_sentence_boundaries():
    paragraph = ' '.join(f'Sentence {i} explains how credit risk changed.' for i in range(60))
    chunks = ch.chunk_text(paragraph, ch.ChunkingConfig(300, 50))

    assert len(chunks) > 1
    assert all(c.startswith('Sentence') and c.endswith('.') for c in chunks)


def test_separators_must_start_with_block_separator():
    with pytest.raises(ValueError):
        ch.ChunkingConfig(1000, 150, ('\n', ' '))


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


def test_join_chunks_drops_repeated_overlap():
    assert ch.join_chunks('Rates rose. Credit held.', 'Credit held. Deposits fell.') == (
        'Rates rose. Credit held. Deposits fell.'
    )
    assert ch.join_chunks('Rates rose.', 'Deposits fell.') == 'Rates rose.\nDeposits fell.'


def test_merge_small_chunks_into_next_then_previous():
    config = ch.ChunkingConfig(50, 0, min_chunk_chars=10)
    chunks = ['Short.', 'A sentence of twenty.', 'x' * 45, 'Tail.']

    assert ch.merge_small_chunks(chunks, config) == [
        'Short.\nA sentence of twenty.', 'x' * 45, 'Tail.',
    ]


def test_merge_small_chunks_never_exceeds_chunk_size():
    config = ch.ChunkingConfig(20, 0, min_chunk_chars=10)
    chunks = ['y' * 18, 'tiny', 'z' * 18]

    assert ch.merge_small_chunks(chunks, config) == chunks


def test_min_chunk_chars_from_config_reduces_short_chunks(tmp_path):
    text = '\n\n'.join(['Heading'] + ['Body sentence about credit. ' * 20] * 3)
    loose = ch.chunk_text(text, ch.ChunkingConfig(300, 0))
    merged = ch.chunk_text(text, ch.ChunkingConfig(300, 0, min_chunk_chars=50))

    assert sum(len(c) < 50 for c in loose) > sum(len(c) < 50 for c in merged)
    assert all(len(c) <= 300 for c in merged)


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
            'heading': '',
            'chunk_index': i,
            'chunk_id': f'GS_2024_mdna_{i:04d}',
        }


def test_chunks_start_with_context_line_and_heading(tmp_path):
    metadata_file = write_section(tmp_path)
    section_file = tmp_path / 'data/processed/sections/GS_2024_mdna.txt'
    section_file.write_text('ASSET & WEALTH MANAGEMENT\n' + TEXT, encoding='utf-8')

    chunks = ch.chunk_section(metadata_file, ch.ChunkingConfig(300, 50), root=tmp_path)

    first, last = chunks[0], chunks[-1]
    assert first.page_content.startswith(
        "Goldman Sachs | FY2024 | Management's Discussion and Analysis | ASSET & WEALTH MANAGEMENT\n"
    )
    assert last.metadata['heading'] == 'ASSET & WEALTH MANAGEMENT'
    assert last.page_content.split('\n', 1)[1].startswith('Line')


def test_context_line_without_heading():
    section = {'company': 'Truist Financial Corp', 'fiscal_year': 2024, 'section_title': 'Risk Factors'}

    assert ch.context_line(section, '') == 'Truist Financial Corp | FY2024 | Risk Factors'


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


def test_min_chunk_chars_must_be_below_chunk_size():
    with pytest.raises(ValueError):
        ch.ChunkingConfig(100, 0, min_chunk_chars=100)
