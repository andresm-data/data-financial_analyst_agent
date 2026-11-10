"""Comprueba que la respuesta del modelo se apoya en una cita real del contexto.

El modelo responde con JSON `{"answer", "quote", "source"}`. La respuesta solo
se acepta si la cita aparece en el chunk `source` (salvo espacios, mayúsculas y
comillas) y si cada cifra de la respuesta aparece en ese chunk. Si no, se
devuelve "I do not know": es mejor no responder que dar una cifra inventada.

Un chunk puede abarcar varios apartados ("Liabilities", luego "Deposits"), así
que la fuente citada lleva el título que está justo antes de la frase citada.
"""
import json
import re

from langchain_core.documents import Document

from financial_analyst_agent.rag.headings import SEPARATOR, is_heading
from financial_analyst_agent.rag.prompt import NO_ANSWER

# =============================================================================
QUOTE_PROBE_CHARS = 50  # inicio de la cita que se busca para ubicarla en el chunk

# Cifras como 13.63, 1,234 o 4.0 (sin el signo $ ni el %).
NUMBER_PATTERN = re.compile(r'\d[\d,]*(?:\.\d+)?')
ELLIPSIS_PATTERN = re.compile(r'\.\.\.|…')
QUOTE_CHARS = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-'})


# =============================================================================
def normalize(text: str) -> str:
    """Minúsculas, comillas y guiones simples y un solo espacio entre palabras."""
    return ' '.join(text.translate(QUOTE_CHARS).lower().split())


def parse_reply(reply: str) -> dict:
    """Lee el JSON del modelo; si no es válido, lo trata como "I do not know"."""
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        return {'answer': NO_ANSWER, 'quote': '', 'source': 0}
    if not isinstance(data, dict):
        return {'answer': NO_ANSWER, 'quote': '', 'source': 0}

    try:
        source = int(data.get('source') or 0)
    except (TypeError, ValueError):
        source = 0

    return {
        'answer': str(data.get('answer') or NO_ANSWER).strip(),
        'quote': str(data.get('quote') or '').strip(),
        'source': source,
    }


def quote_in(quote: str, text: str) -> bool:
    """True si la cita (o cada trozo, si tiene "...") está en el texto."""
    parts = [normalize(part).strip(' ."') for part in ELLIPSIS_PATTERN.split(quote)]
    parts = [part for part in parts if part]
    target = normalize(text)

    return bool(parts) and all(part in target for part in parts)


def numbers_in(answer: str, text: str) -> bool:
    """True si cada cifra de la respuesta aparece en el texto."""
    target = normalize(text)
    return all(number in target for number in NUMBER_PATTERN.findall(answer))


# =============================================================================
def quote_line(text: str, quote: str) -> int | None:
    """Índice de la línea del texto donde empieza (aprox.) la cita, o None."""
    first_part = next((p for p in ELLIPSIS_PATTERN.split(quote) if p.strip()), '')
    probe = normalize(first_part)[:QUOTE_PROBE_CHARS]
    if not probe:
        return None

    seen = ''
    for i, line in enumerate(text.split('\n')):
        seen = f'{seen} {normalize(line)}'
        if probe in seen:
            return i
    return None


def with_quote_heading(doc: Document, quote: str) -> Document:
    """Copia del chunk con el título del apartado donde está la cita.

    Si hay un título dentro del chunk antes de la cita, ese reemplaza al del
    chunk (conservando el título en mayúsculas de nivel superior, si lo hay).
    """
    lines = doc.page_content.split('\n')
    end = quote_line(doc.page_content, quote)
    if end is None:
        return doc

    inner = [
        lines[i].strip() for i in range(end)
        if is_heading(lines[i], lines[i + 1] if i + 1 < len(lines) else None)
    ]
    if not inner:
        return doc

    sub = inner[-1]
    parent = (doc.metadata.get('heading') or '').split(SEPARATOR)[0]
    if sub.isupper() or not parent.isupper() or parent == sub:
        heading = sub
    else:
        heading = f'{parent}{SEPARATOR}{sub}'

    return Document(page_content=doc.page_content, metadata={**doc.metadata, 'heading': heading})


# =============================================================================
def find_source(quote: str, docs: list[Document], source: int) -> Document | None:
    """Chunk que contiene la cita: el que dice el modelo o, si se equivocó, otro."""
    ordered = docs[source - 1:source] if 1 <= source <= len(docs) else []
    for doc in ordered + docs:
        if quote_in(quote, doc.page_content):
            return doc
    return None


def ground(reply: dict, docs: list[Document]) -> dict:
    """Acepta la respuesta solo si su cita y sus cifras están en un chunk.

    Returns:
        dict: `answer`, `quote` y `sources` (el chunk citado, o vacío).
    """
    answer, quote = reply['answer'], reply['quote']
    no_answer = {'answer': NO_ANSWER, 'quote': '', 'sources': []}

    if answer.rstrip('.').lower() == NO_ANSWER.lower() or not quote:
        return no_answer

    doc = find_source(quote, docs, reply['source'])
    if doc is None or not numbers_in(answer, doc.page_content):
        return no_answer

    return {'answer': answer, 'quote': quote, 'sources': [with_quote_heading(doc, quote)]}
