"""Pruebas del chunking de las secciones (sin red ni datos descargados)."""
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
