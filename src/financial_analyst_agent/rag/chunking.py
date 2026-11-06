"""Divide el texto de las secciones de los 10-K en chunks para el RAG.

El tamaño y el solapamiento de los chunks se leen de config/rag.yaml, así que
se pueden ajustar sin tocar el código.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from financial_analyst_agent.ingest.download_10k import PACKAGE_DIR

# =============================================================================
RAG_CONFIG_FILE = PACKAGE_DIR / 'config' / 'rag.yaml'


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
