"""Divide el texto de las secciones de los 10-K en chunks para el RAG.

El tamaño y el solapamiento de los chunks se leen de config/rag.yaml, así que
se pueden ajustar sin tocar el código.

`main` lee las secciones de data/processed/sections/ (un .txt y un .json de
metadatos por sección), las divide y guarda todos los chunks con sus
metadatos en data/processed/chunks/chunks.jsonl.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_analyst_agent.ingest.download_10k import (
    PACKAGE_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from financial_analyst_agent.ingest.sections import SECTIONS_DIR

logger = logging.getLogger(__name__)

# =============================================================================
RAG_CONFIG_FILE = PACKAGE_DIR / 'config' / 'rag.yaml'
CHUNKS_DIR = PROCESSED_DIR / 'chunks'
CHUNKS_FILE = CHUNKS_DIR / 'chunks.jsonl'

# Metadatos de la sección que se copian a cada chunk.
CHUNK_METADATA_KEYS = [
    'company', 'ticker', 'fiscal_year', 'filing_date',
    'section', 'section_title', 'source_file', 'section_file',
]


# =============================================================================
@dataclass(frozen=True)
class ChunkingConfig:
    """Parámetros del divisor de texto, en caracteres."""

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError('chunk_size debe ser mayor que 0')
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError('chunk_overlap debe estar entre 0 y chunk_size')


# =============================================================================
def load_chunking_config(path: Path = RAG_CONFIG_FILE) -> ChunkingConfig:
    """Lee la sección `chunking` de config/rag.yaml."""
    with path.open(encoding='utf-8') as f:
        chunking = yaml.safe_load(f)['chunking']

    return ChunkingConfig(
        chunk_size=int(chunking['chunk_size']),
        chunk_overlap=int(chunking['chunk_overlap']),
    )


# =============================================================================
def make_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """Divisor que corta primero por párrafos, luego líneas, luego palabras."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


# =============================================================================
def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """Divide un texto en chunks según la configuración.

    Args:
        text (str): Texto a dividir.
        config (ChunkingConfig | None): Parámetros; si es None se leen de
            config/rag.yaml.

    Returns:
        list[str]: Chunks de como máximo `chunk_size` caracteres.
    """
    config = config or load_chunking_config()

    return make_splitter(config).split_text(text)


# =============================================================================
def chunk_section(
    metadata_file: Path, config: ChunkingConfig, root: Path = PROJECT_ROOT
) -> list[Document]:
    """Divide una sección y le pone a cada chunk los metadatos de su JSON.

    Args:
        metadata_file (Path): JSON de la sección generado por `sections`.
        config (ChunkingConfig): Parámetros del divisor.
        root (Path): Raíz contra la que se resuelve `section_file`.

    Returns:
        list[Document]: Chunks con `company`, `ticker`, `fiscal_year`,
            `filing_date`, `section`, `section_title`, `source_file`,
            `section_file`, `chunk_index` y `chunk_id`.
    """
    section = json.loads(metadata_file.read_text(encoding='utf-8'))
    text = (root / section['section_file']).read_text(encoding='utf-8')
    base = {key: section[key] for key in CHUNK_METADATA_KEYS}
    stem = f"{section['ticker']}_{section['fiscal_year']}_{section['section']}"

    return [
        Document(
            page_content=chunk,
            metadata={**base, 'chunk_index': i, 'chunk_id': f'{stem}_{i:04d}'},
        )
        for i, chunk in enumerate(chunk_text(text, config))
    ]


# =============================================================================
def save_chunks(chunks: list[Document], path: Path = CHUNKS_FILE) -> None:
    """Guarda los chunks en JSONL: una línea con `text` y `metadata` por chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(
                {'text': chunk.page_content, 'metadata': chunk.metadata},
                ensure_ascii=False,
            ) + '\n')


# =============================================================================
def load_chunks(path: Path = CHUNKS_FILE) -> list[Document]:
    """Lee los chunks guardados por `save_chunks`."""
    with path.open(encoding='utf-8') as f:
        return [
            Document(page_content=row['text'], metadata=row['metadata'])
            for row in map(json.loads, f)
        ]


# =============================================================================
def chunk_counts(chunks: list[Document]) -> pd.DataFrame:
    """Tabla de chunks por banco y sección, con total por banco y general."""
    counts = (
        pd.DataFrame([c.metadata for c in chunks])
        .pivot_table(
            index=['ticker', 'company'], columns='section',
            values='chunk_id', aggfunc='count', fill_value=0,
        )
        .reindex(columns=['risk_factors', 'mdna'], fill_value=0)
    )
    counts['total'] = counts.sum(axis=1)
    counts.loc[('TOTAL', ''), :] = counts.sum()
    counts.columns.name = None

    return counts.astype(int).reset_index()


# =============================================================================
def main() -> None:
    """Divide todas las secciones, guarda chunks.jsonl e imprime el conteo.

    Raises:
        FileNotFoundError: Si no hay JSON de secciones.
    """
    metadata_files = sorted(SECTIONS_DIR.glob('*_*_*.json'))

    if not metadata_files:
        raise FileNotFoundError(f'No hay secciones en {SECTIONS_DIR}')

    config = load_chunking_config()
    chunks = [
        chunk for metadata_file in metadata_files
        for chunk in chunk_section(metadata_file, config)
    ]
    save_chunks(chunks)

    logger.info(
        '%d secciones -> %d chunks (chunk_size=%d, chunk_overlap=%d) en %s',
        len(metadata_files), len(chunks), config.chunk_size,
        config.chunk_overlap, CHUNKS_FILE,
    )
    print(chunk_counts(chunks).to_string(index=False))


# =============================================================================
def run() -> None:
    """Configura los logs y ejecuta main; es el punto de entrada del comando."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    main()


# =============================================================================
if __name__ == '__main__':
    run()
