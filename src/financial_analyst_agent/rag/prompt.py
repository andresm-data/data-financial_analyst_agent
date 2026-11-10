"""Prompt del RAG: responder solo con el contexto recuperado de los 10-K.

Si el contexto no tiene la respuesta, el modelo debe decir exactamente
`NO_ANSWER` ("I do not know") en vez de inventar una cifra.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
NO_ANSWER = 'I do not know'

SYSTEM_RULES = f"""You are a financial analyst who answers questions about banks' 10-K annual reports.

Rules:
1. Answer ONLY with information from the context below. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly "{NO_ANSWER}" and nothing else.
3. Do not guess, estimate or calculate figures that are not written in the context.
4. Copy numbers, units and periods exactly as they appear in the context.
5. Answer in at most 4 sentences."""

HUMAN_TEMPLATE = """Context:
{context}

Question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_RULES),
    ('human', HUMAN_TEMPLATE),
])

# El modelo copia la frase que sostiene la respuesta y dice de qué chunk sale,
# para comprobar en código que la cita existe (ver `rag/grounding.py`).
GROUNDED_RULES = f"""You are a financial analyst who answers questions about banks' 10-K annual reports.

Rules:
1. Answer ONLY with information from the numbered context below. Do not use outside knowledge.
2. Find the sentence or table row that answers the question and copy it word for word into "quote".
3. Write in "source" the number [n] of the context piece that contains the quote.
4. Write "answer" in at most 3 sentences, using only the quote. Copy numbers and units exactly.
5. If no piece of the context answers the question, reply {{{{"answer": "{NO_ANSWER}", "quote": "", "source": 0}}}}.

Reply only with a JSON object with the keys "answer", "quote" and "source"."""

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ('system', GROUNDED_RULES),
    ('human', HUMAN_TEMPLATE),
])


# =============================================================================
def source_label(doc: Document) -> str:
    """Empresa, año fiscal y sección de un chunk ("JPMorgan Chase | FY2024 | ...")."""
    meta = doc.metadata
    section = meta.get('section_title') or meta.get('section')

    return f"{meta.get('company')} | FY{meta.get('fiscal_year')} | {section}"


def format_docs(docs: list[Document]) -> str:
    """Une los chunks en un solo contexto, cada uno con su fuente como encabezado."""
    return '\n\n'.join(
        f'[{n}] {source_label(doc)}\n{doc.page_content}' for n, doc in enumerate(docs, 1)
    )
