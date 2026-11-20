"""Cadena text-to-SQL: prompt -> llama3.2:3b (temperatura 0) -> SQL limpio.

Solo genera el SQL; no lo ejecuta (eso lo hace la herramienta en `sql/tool.py`).

Uso: `poetry run python -m financial_analyst_agent.sql.chain "pregunta"`.
"""
import re
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.sql.prompt import build_sql_prompt, build_sql_repair_prompt

# =============================================================================
# Bloque ```sql ... ``` (o ``` ... ```); se queda con lo de dentro.
FENCE_RE = re.compile(r'```[a-zA-Z]*\s*\n?(.*?)```', re.DOTALL)

DEMO_QUESTION = "What was Goldman Sachs' net income in fiscal year 2021?"


# =============================================================================
def clean_sql(text: str) -> str:
    """Quita las marcas de markdown y los espacios sobrantes de la salida.

    Si el modelo envuelve el SQL en ```sql ... ```, se toma el primer bloque;
    si no, se devuelve el texto tal cual, sin espacios al inicio ni al final.
    """
    match = FENCE_RE.search(text)
    if match:
        text = match.group(1)
    return text.strip()


def build_sql_chain(llm=None) -> Runnable:
    """LCEL: {'question': ...} -> prompt -> modelo -> texto -> SQL limpio."""
    llm = llm or get_llm(temperature=0.0)
    return build_sql_prompt() | llm | StrOutputParser() | RunnableLambda(clean_sql)


def build_sql_repair_chain(llm=None) -> Runnable:
    """LCEL: {'question', 'sql', 'error'} -> prompt de corrección -> SQL limpio."""
    llm = llm or get_llm(temperature=0.0)
    return build_sql_repair_prompt() | llm | StrOutputParser() | RunnableLambda(clean_sql)


def main() -> None:
    """Genera (sin ejecutar) el SQL de una pregunta y lo imprime."""
    question = ' '.join(sys.argv[1:]) or DEMO_QUESTION
    sql = build_sql_chain().invoke({'question': question})

    print(f'Pregunta: {question}')
    print(f'SQL:\n{sql}')


# =============================================================================
if __name__ == '__main__':
    main()
