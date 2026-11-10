"""Pruebas de la cadena RAG (retriever y modelo falsos, sin Ollama)."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from financial_analyst_agent.rag import chain as ch
from financial_analyst_agent.rag import vectorstore as vs

DOC = Document(
    page_content='Net interest income was $92.6 billion.',
    metadata={'company': 'JPMorgan Chase', 'ticker': 'JPM', 'fiscal_year': 2024,
              'section': 'mdna', 'section_title': "Management's Discussion and Analysis"},
)


class RecordingModel(FakeListChatModel):
    """Modelo falso que guarda los mensajes que recibe."""

    seen: list = []

    def _call(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._call(messages, *args, **kwargs)


def test_chain_returns_answer_and_sources():
    llm = RecordingModel(responses=['It was $92.6 billion.'], seen=[])
    chain = ch.build_chain(RunnableLambda(lambda q: [DOC]), llm)

    result = chain.invoke('What was JPMorgan net interest income in 2024?')

    assert result['answer'] == 'It was $92.6 billion.'
    assert result['context'] == [DOC]
    assert result['question'] == 'What was JPMorgan net interest income in 2024?'
    human = llm.seen[0][-1].content
    assert '[1] JPMorgan Chase | FY2024' in human
    assert '$92.6 billion' in human


def test_chain_runs_with_empty_context():
    chain = ch.build_chain(RunnableLambda(lambda q: []), FakeListChatModel(responses=['I do not know']))

    result = chain.invoke('Anything?')

    assert result == {'context': [], 'question': 'Anything?', 'answer': 'I do not know'}


def test_retriever_filters_by_bank_and_year(tmp_path):
    store = vs.get_vectorstore(tmp_path, DeterministicFakeEmbedding(size=8))
    other = Document(page_content='x', metadata={**DOC.metadata, 'fiscal_year': 2023})
    store.add_documents([DOC, other], ids=['a', 'b'])

    docs = ch.get_retriever(store, k=5).invoke('JPMorgan income in 2024')

    assert [d.metadata['fiscal_year'] for d in docs] == [2024]


BAC = Document(page_content='y', metadata={'company': 'Bank of America', 'fiscal_year': 2023,
                                           'section': 'risk_factors', 'section_title': 'Risk Factors'})


def test_citations_remove_duplicates_in_order():
    assert ch.citations([DOC, BAC, DOC]) == [
        "JPMorgan Chase | FY2024 | Management's Discussion and Analysis",
        'Bank of America | FY2023 | Risk Factors',
    ]


def test_format_answer_lists_company_year_and_section():
    text = ch.format_answer({'question': 'q?', 'answer': ' $92.6 billion. ', 'context': [DOC, BAC]})

    assert text == (
        'Q: q?\nA: $92.6 billion.\nSources:\n'
        "  - JPMorgan Chase | FY2024 | Management's Discussion and Analysis\n"
        '  - Bank of America | FY2023 | Risk Factors'
    )


def test_format_answer_without_answer_has_no_sources():
    text = ch.format_answer({'question': 'q?', 'answer': 'I do not know.', 'context': [DOC]})

    assert text.endswith('Sources: none (the answer is not in the context)')
    assert 'JPMorgan' not in text


def test_is_no_answer():
    assert ch.is_no_answer('I do not know')
    assert ch.is_no_answer(' i do not know. ')
    assert ch.is_no_answer('The context does not provide a reason. However, in 2025...')
    assert not ch.is_no_answer('I do not know the exact figure, but it rose.')


def test_clean_answer():
    assert ch.clean_answer(' I do not know. ') == 'I do not know'
    assert ch.clean_answer(' $92.6 billion. ') == '$92.6 billion.'


def test_chain_returns_exact_no_answer():
    chain = ch.build_chain(RunnableLambda(lambda q: []), FakeListChatModel(responses=['I do not know.']))

    assert chain.invoke('Anything?')['answer'] == 'I do not know'
