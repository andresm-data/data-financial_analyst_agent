"""Pruebas de la detección de tablas y del chunking que no las parte."""
from financial_analyst_agent.rag import chunking as ch
from financial_analyst_agent.rag import tables as tb

PROSE = 'Net income rose because revenue grew and expenses were flat. ' * 3

TABLE = [
    'Table 1 Summary Income Statement',
    '(Dollars in millions) 2024 2023',
    'Net interest income $ 56,060 $ 56,931',
    'Per common share information',
    'Earnings $ 3.25 $ 3.10',
    'Efficiency ratio 65.57 % 66.79 %',
]


def test_is_row_and_is_label():
    assert tb.is_row('Noninterest income 45,827 41,650')
    assert tb.is_row('Provisions (165) (549)')
    assert not tb.is_row('Net income was $27.1 billion in 2024.')
    assert tb.is_label('Per common share information')
    assert not tb.is_label(PROSE)


def test_table_includes_title_and_stops_before_prose():
    lines = [PROSE, *TABLE, PROSE]

    assert tb.table_ranges(lines) == [(1, 7)]


def test_no_chunk_starts_in_the_middle_of_a_table():
    rows = [f'Loan category {i} $ {i},000 $ {i},500' for i in range(80)]
    text = '\n'.join([PROSE] * 5 + TABLE[:2] + rows + [PROSE] * 5)
    chunks = ch.chunk_text(text, ch.ChunkingConfig(400, 60))
    header = TABLE[0] + ch.CONTINUED

    starts_mid_table = [
        c for c in chunks if c.split('\n', 1)[0] in rows
    ]
    assert starts_mid_table == []
    assert all(len(c) <= 400 for c in chunks)
    assert sum(c.startswith(header) for c in chunks) > 1


def test_intro_sentence_moves_to_its_table():
    lines = ['Fees grew this year. The table below presents fee income by business.', *TABLE]
    blocks = tb.split_blocks('\n'.join(lines))

    assert [b.lines for b in blocks] == [
        ['Fees grew this year.'],
        ['The table below presents fee income by business.', *TABLE],
    ]
    assert blocks[1].intro == 1


def test_intro_sentences_that_do_not_introduce_the_next_table_stay():
    for line in ['In the table above, revenue rose by business.', PROSE]:
        blocks = tb.split_blocks('\n'.join([line, *TABLE]))

        assert blocks[0].lines == [line]
        assert blocks[1].intro == 0


def test_table_header_keeps_columns_when_intro_does_not_fit():
    intro = 'The following table presents ' + 'a long description ' * 10 + ':'
    block = tb.Block([intro, *TABLE], is_table=True, intro=1)

    assert tb.table_header(block, max_chars=100) == TABLE[:2]
    assert tb.table_header(block, max_chars=1000) == [intro, *TABLE[:2]]
