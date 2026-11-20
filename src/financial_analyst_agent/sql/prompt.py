"""Prompt text-to-SQL: pregunta en inglés -> una consulta SELECT de DuckDB.

El esquema y los nombres de métricas no se copian aquí: se leen al ejecutar de
`docs/schema.md` y `config/metrics.yaml`, que son la única fuente de verdad.

Los 3 ejemplos se ejecutaron sobre `finance.duckdb` y devuelven filas. No son
preguntas de `eval/questions.csv`, para no inflar la medición de precisión.
"""
from functools import cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

from financial_analyst_agent.ingest.download_10k import PACKAGE_DIR, PROJECT_ROOT

# =============================================================================
SCHEMA_FILE = PROJECT_ROOT / 'docs' / 'schema.md'
METRICS_FILE = PACKAGE_DIR / 'config' / 'metrics.yaml'

# (pregunta, SQL) de ejemplo: consulta directa, ranking y variación porcentual.
EXAMPLES = [
    (
        "What were Morgan Stanley's total liabilities at the end of fiscal year 2021?",
        """SELECT company, fiscal_year, value AS liabilities_usd
FROM financials
WHERE company = 'MS' AND metric = 'Liabilities' AND fiscal_year = 2021""",
    ),
    (
        'Which bank had the highest net income in fiscal year 2018?',
        """SELECT company, value AS net_income_usd
FROM financials
WHERE metric = 'NetIncomeLoss' AND fiscal_year = 2018
ORDER BY value DESC
LIMIT 1""",
    ),
    (
        "By what percent did Citigroup's total deposits change from fiscal year 2016 to 2019?",
        """SELECT
    MAX(CASE WHEN fiscal_year = 2016 THEN value END) AS deposits_2016_usd,
    MAX(CASE WHEN fiscal_year = 2019 THEN value END) AS deposits_2019_usd,
    (MAX(CASE WHEN fiscal_year = 2019 THEN value END)
     / MAX(CASE WHEN fiscal_year = 2016 THEN value END) - 1) * 100 AS pct_change
FROM financials
WHERE company = 'C' AND metric = 'Deposits' AND fiscal_year IN (2016, 2019)""",
    ),
]

SYSTEM_RULES = """You write DuckDB SQL queries that answer questions about the annual figures of US banks.

{schema}

## Metric names
Use exactly one of these values in `metric`. Everyday words map to them like this:
{metrics}

## Rules
1. Write exactly ONE SQL query, and it must be a SELECT statement.
2. Use only the table and columns described above.
3. Filter `company` by ticker and `metric` by its exact name from the list above.
4. Return the raw `value` in US dollars; give every computed column a clear alias.
5. Reply with the SQL only: no explanation, no comments, no markdown."""

REPAIR_REQUEST = """Question: {question}

This SQL query failed:
{sql}

Error message:
{error}

Write a corrected DuckDB SELECT query that answers the question. Follow the
rules above and reply with the SQL only."""


# =============================================================================
@cache
def load_schema(path: Path = SCHEMA_FILE) -> str:
    """Texto completo de `docs/schema.md`."""
    return path.read_text(encoding='utf-8').strip()


@cache
def load_metric_names(path: Path = METRICS_FILE) -> dict[str, list[str]]:
    """Nombre exacto de cada métrica -> palabras cotidianas que la nombran."""
    return yaml.safe_load(path.read_text(encoding='utf-8'))['metrics']


def format_metrics(metrics: dict[str, list[str]]) -> str:
    """Una línea por métrica: `- NetIncomeLoss: profit, net income, ...`."""
    return '\n'.join(f"- {name}: {', '.join(words)}" for name, words in metrics.items())


def build_sql_prompt() -> ChatPromptTemplate:
    """Prompt con el esquema, las métricas y los ejemplos; espera `{question}`.

    El esquema y las métricas entran como valores parciales, así que sus
    llaves o símbolos no se interpretan como variables de la plantilla.
    """
    messages = [('system', SYSTEM_RULES)]
    for question, sql in EXAMPLES:
        messages += [('human', question), ('ai', sql)]
    messages.append(('human', '{question}'))

    return ChatPromptTemplate.from_messages(messages).partial(
        schema=load_schema(), metrics=format_metrics(load_metric_names()),
    )


def build_sql_repair_prompt() -> ChatPromptTemplate:
    """Prompt para corregir un SQL que falló; espera `{question}`, `{sql}` y `{error}`.

    Lleva las mismas reglas, esquema y métricas que `build_sql_prompt`.
    """
    return ChatPromptTemplate.from_messages(
        [('system', SYSTEM_RULES), ('human', REPAIR_REQUEST)],
    ).partial(schema=load_schema(), metrics=format_metrics(load_metric_names()))
