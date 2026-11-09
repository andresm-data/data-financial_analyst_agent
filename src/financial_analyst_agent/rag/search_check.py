"""Comprueba la búsqueda en Chroma con 3 preguntas sobre bancos concretos.

Para cada pregunta hace `similarity_search` con k=5, filtrando por el banco
y el año que nombra la pregunta (`retrieval.search`), imprime
los resultados con su metadata y comprueba que cada chunk viene del banco y
del año fiscal esperados y que tiene la metadata completa. Uso:
`poetry run search_check` (después de `index_chunks`).
"""
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from financial_analyst_agent.rag.chunking import CHUNK_METADATA_KEYS
from financial_analyst_agent.rag.retrieval import search
from financial_analyst_agent.rag.vectorstore import get_vectorstore

# =============================================================================
K = 5
REQUIRED_METADATA = [*CHUNK_METADATA_KEYS, 'chunk_index', 'chunk_id']


@dataclass(frozen=True)
class Question:
    """Pregunta de prueba y de qué 10-K deberían salir los chunks."""

    text: str
    ticker: str
    fiscal_year: int


QUESTIONS = [
    Question("What drove JPMorgan Chase's net interest income in 2024?", 'JPM', 2024),
    Question('How did Bank of America deposits change in 2023?', 'BAC', 2023),
    Question('What risks does Goldman Sachs see from its Asset & Wealth Management '
             'business in 2025?', 'GS', 2025),
]


# =============================================================================
def problems(doc: Document, question: Question) -> list[str]:
    """Motivos por los que un resultado no es el esperado (vacío si está bien)."""
    meta = doc.metadata
    found = [
        f'falta {key}' for key in REQUIRED_METADATA
        if meta.get(key) in (None, '')
    ]
    if meta.get('ticker') != question.ticker:
        found.append(f"banco {meta.get('ticker')} != {question.ticker}")
    if meta.get('fiscal_year') != question.fiscal_year:
        found.append(f"año {meta.get('fiscal_year')} != {question.fiscal_year}")

    return found


# =============================================================================
def check(store: Chroma, questions: list[Question] = QUESTIONS, k: int = K) -> int:
    """Imprime los resultados de cada pregunta y devuelve cuántos fallan."""
    failures = 0

    for n, question in enumerate(questions, 1):
        print(f'\n[{n}] {question.text}  (esperado: {question.ticker} {question.fiscal_year})')
        for rank, doc in enumerate(search(store, question.text, k=k), 1):
            meta = doc.metadata
            issues = problems(doc, question)
            failures += bool(issues)
            status = 'OK ' if not issues else 'ERR'
            snippet = ' '.join(doc.page_content.split())[:90]
            print(
                f"  {rank}. {status} {meta.get('company')} | FY{meta.get('fiscal_year')} | "
                f"{meta.get('section')} | {meta.get('chunk_id')}"
            )
            print(f'       {snippet}')
            if issues:
                print(f"       -> {'; '.join(issues)}")

    total = len(questions) * k
    print(f'\n{total - failures}/{total} resultados del banco y año correctos')
    return failures


# =============================================================================
def main() -> None:
    """Abre la colección persistente y ejecuta la comprobación."""
    store = get_vectorstore()
    if store._collection.count() == 0:
        raise RuntimeError('La colección está vacía; ejecuta `index_chunks` antes')

    raise SystemExit(1 if check(store) else 0)


# =============================================================================
if __name__ == '__main__':
    main()
