"""Búsqueda en Chroma filtrada por el banco y el año que nombra la pregunta.

La búsqueda semántica sola no distingue bancos ni años: el texto de los
chunks casi nunca dice de qué banco es, y los 10-K de años distintos repiten
párrafos casi iguales. Por eso se buscan en la pregunta los bancos (por
nombre, alias o ticker de config/companies.yaml) y los años, y se usan como
filtro sobre la metadata `ticker` y `fiscal_year` antes de comparar vectores.
"""
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from financial_analyst_agent.ingest.download_10k import load_banks

# =============================================================================
K = 5

# Años de cuatro cifras sueltos o con prefijo FY ("2024", "FY2024").
YEAR_PATTERN = re.compile(r'(?<!\d)(?:FY)?(20\d{2})(?!\d)', re.IGNORECASE)


# =============================================================================
def bank_patterns(banks: list[dict] | None = None) -> list[tuple[re.Pattern, str]]:
    """Patrones que reconocen a cada banco en una pregunta, con su ticker.

    El nombre y los alias se buscan sin distinguir mayúsculas; los alias en
    mayúsculas (tickers como "GS") solo cuentan escritos en mayúsculas, para
    no confundirlos con palabras comunes ("ms", "gs").
    """
    patterns = []
    for bank in banks if banks is not None else load_banks():
        for alias in [bank['name'], *bank.get('aliases', [])]:
            flags = 0 if alias.isupper() else re.IGNORECASE
            patterns.append((re.compile(rf'(?<!\w){re.escape(alias)}(?!\w)', flags), bank['ticker']))

    return patterns


# =============================================================================
def find_tickers(question: str, patterns: list[tuple[re.Pattern, str]]) -> list[str]:
    """Tickers de los bancos nombrados en la pregunta, sin repetir."""
    return sorted({ticker for pattern, ticker in patterns if pattern.search(question)})


def find_years(question: str) -> list[int]:
    """Años fiscales nombrados en la pregunta, sin repetir."""
    return sorted({int(year) for year in YEAR_PATTERN.findall(question)})


# =============================================================================
def build_filter(tickers: list[str], years: list[int]) -> dict | None:
    """Filtro de Chroma sobre `ticker` y `fiscal_year` (None si no hay nada)."""
    conditions = []
    if tickers:
        conditions.append({'ticker': {'$in': tickers}})
    if years:
        conditions.append({'fiscal_year': {'$in': years}})

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {'$and': conditions}


def question_filter(question: str, banks: list[dict] | None = None) -> dict | None:
    """Filtro de metadata para los bancos y años que nombra la pregunta."""
    return build_filter(find_tickers(question, bank_patterns(banks)), find_years(question))


# =============================================================================
def search(
    store: Chroma, question: str, k: int = K, banks: list[dict] | None = None
) -> list[Document]:
    """Los `k` chunks más parecidos a la pregunta, del banco y año que nombra.

    Si la pregunta no nombra banco ni año, la búsqueda no se filtra. Si nombra
    un año que no está indexado, no devuelve nada.
    """
    return store.similarity_search(question, k=k, filter=question_filter(question, banks))
