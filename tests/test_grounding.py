"""Pruebas de la verificación de citas y de la cadena con cita (sin Ollama)."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from financial_analyst_agent.rag import chain as ch
from financial_analyst_agent.rag import grounding as gr

CITI = Document(
    page_content='Basel III ratios\nCET1 Capital(4)\n13.63 % 13.37 %\nCitigroup’s Supplementary  Leverage ratio was 5.8%.',
    metadata={'chunk_id': 'C_0054', 'company': 'Citigroup', 'fiscal_year': 2024, 'section': 'mdna',
              'section_title': "Management's Discussion and Analysis"},
)
OTHER = Document(page_content='Deposits decreased $6.5 billion.',
                 metadata={'chunk_id': 'BAC_0035', 'company': 'Bank of America', 'fiscal_year': 2023,
                           'section': 'mdna', 'section_title': "Management's Discussion and Analysis"})


def reply(answer, quote, source):
    return {'answer': answer, 'quote': quote, 'source': source}


@pytest.mark.parametrize('raw, expected', [
    ('{"answer": "13.63%", "quote": "CET1 Capital 13.63 %", "source": "1"}', reply('13.63%', 'CET1 Capital 13.63 %', 1)),
    ('not json', reply('I do not know', '', 0)),
    ('[1, 2]', reply('I do not know', '', 0)),
    ('{"answer": "x", "quote": "y", "source": "one"}', reply('x', 'y', 0)),
])
def test_parse_reply(raw, expected):
    assert gr.parse_reply(raw) == expected


def test_quote_in_ignores_spaces_case_curly_quotes_and_ellipsis():
    assert gr.quote_in("citigroup's supplementary leverage ratio was 5.8%", CITI.page_content)
    assert gr.quote_in('CET1 Capital(4) 13.63 %', CITI.page_content)
    assert gr.quote_in('Basel III ratios ... Leverage ratio was 5.8%.', CITI.page_content)
    assert not gr.quote_in('CET1 Capital was 14.0%', CITI.page_content)
    assert not gr.quote_in(' ... ', CITI.page_content)


def test_numbers_in():
    assert gr.numbers_in('The CET1 ratio was 13.63%.', CITI.page_content)
    assert not gr.numbers_in('The CET1 ratio was 14.1%.', CITI.page_content)


def test_ground_accepts_quote_from_the_cited_or_another_chunk():
    ok = gr.ground(reply('CET1 was 13.63%.', 'CET1 Capital(4) 13.63 %', 2), [OTHER, CITI])
    wrong_source = gr.ground(reply('CET1 was 13.63%.', 'CET1 Capital(4) 13.63 %', 1), [OTHER, CITI])

    assert ok == {'answer': 'CET1 was 13.63%.', 'quote': 'CET1 Capital(4) 13.63 %', 'sources': [CITI]}
    assert wrong_source['sources'] == [CITI]


@pytest.mark.parametrize('bad', [
    reply('CET1 was 13.63%.', 'CET1 was 13.63% according to Citi', 1),  # cita inventada
    reply('CET1 was 14.5%.', 'CET1 Capital(4) 13.63 %', 1),             # cifra que no está
    reply('CET1 was 13.63%.', '', 1),                                    # sin cita
    reply('I do not know.', 'CET1 Capital(4) 13.63 %', 1),
])
def test_ground_rejects_unsupported_answers(bad):
    assert gr.ground(bad, [CITI]) == {'answer': 'I do not know', 'quote': '', 'sources': []}


def test_grounded_chain_cites_only_the_used_source():
    raw = json.dumps({'answer': 'CET1 was 13.63%.', 'quote': 'CET1 Capital(4) 13.63 %', 'source': 2})
    chain = ch.build_grounded_chain(RunnableLambda(lambda q: [OTHER, CITI]), FakeListChatModel(responses=[raw]))

    result = chain.invoke('Citi CET1 2024?')
    text = ch.format_answer(result)

    assert result['answer'] == 'CET1 was 13.63%.'
    assert result['sources'] == [CITI]
    assert 'Quote: "CET1 Capital(4) 13.63 %"' in text
    assert 'Citigroup | FY2024' in text
    assert 'Bank of America' not in text


def test_grounded_chain_turns_bad_json_into_no_answer():
    chain = ch.build_grounded_chain(RunnableLambda(lambda q: [CITI]), FakeListChatModel(responses=['oops']))

    result = chain.invoke('q?')

    assert result['answer'] == 'I do not know'
    assert ch.format_answer(result).endswith('Sources: none (the answer is not in the context)')
