"""Pruebas del nodo `answer` con un modelo falso."""
from langchain_core.messages import AIMessage

from financial_analyst_agent.agent import answer as a
from financial_analyst_agent.rag.prompt import NO_ANSWER, SYSTEM_RULES

SOURCE = {'company': 'Citigroup', 'fiscal_year': 2024, 'section': 'Risk Factors',
          'source_file': 'f.htm', 'label': 'Citigroup | FY2024 | Risk Factors', 'text': 'Rates may fall.'}


class FakeLLM:
    def __init__(self, reply='ok'):
        self.reply, self.calls = reply, []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(self.reply, usage_metadata={'input_tokens': 100, 'output_tokens': 5, 'total_tokens': 105})


def test_prompt_reuses_rag_rules_unchanged():
    assert a.ANSWER_RULES.startswith(SYSTEM_RULES)
    assert 'US dollars' in a.ANSWER_RULES and '(financials table)' in a.ANSWER_RULES
    assert 'report the value from the SQL result' in a.ANSWER_RULES


def test_no_context_does_not_call_model():
    llm = FakeLLM()
    assert a.answer({'question': 'q', 'sources': [], 'rows': []}, llm=llm)['answer'] == a.NO_CONTEXT
    assert a.answer({'question': 'q', 'sql': 'SELECT 1', 'rows': []}, llm=llm)['answer'] == a.NO_ROWS
    failed = a.answer({'question': 'q', 'rows': [], 'error': 'SQL error: boom'}, llm=llm)['answer']
    assert failed.startswith(NO_ANSWER) and 'SQL error: boom' in failed
    assert llm.calls == []


def test_one_call_with_text_and_rows():
    llm = FakeLLM('Net income was USD 1.5 billion (Citigroup, FY2024, Risk Factors).')
    state = {'question': 'q', 'sources': [SOURCE], 'sql': 'SELECT x', 'rows': [{'net_income_usd_bn': 1.5}]}
    assert a.answer(state, llm=llm)['answer'].startswith('Net income')
    assert len(llm.calls) == 1
    human = llm.calls[0][1].content
    assert '[1] Citigroup | FY2024 | Risk Factors' in human and '{"net_income_usd_bn": 1.5}' in human
    assert 'SQL: SELECT x' in human


def test_sql_error_with_text_adds_note():
    out = a.answer({'question': 'q', 'sources': [SOURCE], 'rows': [], 'error': 'SQL error: boom'}, llm=FakeLLM())
    assert out['answer'].endswith('SQL error: boom')


def test_no_answer_is_normalized():
    out = a.answer({'question': 'q', 'sources': [SOURCE]}, llm=FakeLLM('The context does not say.'))
    assert out['answer'] == NO_ANSWER


def test_sources_block_is_added_from_state():
    state = {'question': 'q', 'sources': [SOURCE, SOURCE], 'sql': 'SELECT x', 'rows': [{'x': 1}]}
    out = a.answer(state, llm=FakeLLM('Some answer.'))['answer']
    assert out.endswith('Sources:\n  - Citigroup | FY2024 | Risk Factors\n'
                        '  - financials table (SQL query above; amounts in US dollars)')
    assert 'Sources:' not in a.answer({'question': 'q', 'sources': [SOURCE]}, llm=FakeLLM(NO_ANSWER))['answer']


def test_rows_are_capped_in_prompt_and_answer_says_so():
    rows = [{'company': f'B{i}', 'value_usd_bn': i} for i in range(a.MAX_PROMPT_ROWS + 5)]
    llm = FakeLLM('Some answer.')
    out = a.answer({'question': 'q', 'sql': 'SELECT x', 'rows': rows}, llm=llm)['answer']
    human = llm.calls[0][1].content
    assert f'"company": "B{a.MAX_PROMPT_ROWS - 1}"' in human
    assert f'"company": "B{a.MAX_PROMPT_ROWS}"' not in human
    assert f'only the first {a.MAX_PROMPT_ROWS} of {len(rows)} rows' in human
    assert out.endswith(f'first {a.MAX_PROMPT_ROWS} of {len(rows)} rows returned by the SQL query.')


def test_no_truncation_note_for_few_rows():
    out = a.answer({'question': 'q', 'sql': 'SELECT x', 'rows': [{'x': 1}]}, llm=FakeLLM('ok'))['answer']
    assert 'Note:' not in out


def test_table_figure_is_added_when_text_and_rows():
    state = {'question': 'q', 'sources': [SOURCE], 'sql': 'SELECT x', 'rows': [{'net_income_usd_bn': 58.471}]}
    out = a.answer(state, llm=FakeLLM('Net income was $58.5 billion.'))['answer']
    assert '  - net_income_usd_bn = 58.471' in out
    assert out.index('Figures from the financials table') < out.index('Sources:')
    only_rows = a.answer({'question': 'q', 'sql': 'SELECT x', 'rows': [{'x': 1}]}, llm=FakeLLM('ok'))['answer']
    assert 'Figures from the financials table' not in only_rows


def test_table_figures_caps_rows():
    text = a.table_figures([{'x': i} for i in range(8)])
    assert '  - x = 4' in text and '  - x = 5' not in text and '... and 3 more rows' in text
