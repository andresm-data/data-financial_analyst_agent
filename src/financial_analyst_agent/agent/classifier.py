"""Nodo `classify` del agente: decide qué tipo de pregunta hizo el usuario.

Devuelve una de las rutas de `ROUTES` (`text`, `numbers`, `both`,
`out_of_scope`) y una frase con el motivo. Solo decide el tipo de pregunta; no
comprueba si el banco o el año existen en los datos (eso lo resuelven los
nodos de RAG y SQL).

Las métricas de la tabla se leen de `config/metrics.yaml` al construir el
prompt. Los 8 ejemplos (2 por ruta) son inventados y no pueden coincidir con
`eval/questions.csv`; `tests/test_classifier.py` lo comprueba por código.

Uso rápido: `poetry run python -m financial_analyst_agent.agent.classifier "pregunta"`
"""
import sys
from functools import cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from financial_analyst_agent.agent.schemas import ROUTES, RouteDecision
from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.sql.prompt import format_metrics, load_metric_names

# Método de `with_structured_output` que funciona con llama3.2:3b en Ollama.
STRUCTURED_METHOD = 'json_schema'

# (pregunta, ruta, motivo). Bancos, años y métricas distintos de eval/questions.csv.
EXAMPLES: list[tuple[str, str, str]] = [
    ('What does Wells Fargo say about the risks of relying on third-party vendors in its 2021 annual report?',
     'text', 'It asks about risks described in the narrative of the report.'),
    ('How does U.S. Bancorp describe its strategy for growing its payments business in its 2019 report?',
     'text', 'It asks for a description of strategy from the narrative, not for a figure.'),
    ('Rank every bank by revenue for 2018, from highest to lowest.',
     'numbers', 'It asks for a ranking of a metric from the table of figures.'),
    ('How much net income did U.S. Bancorp report for 2016?',
     'numbers', 'It asks for the value of one metric for one bank and year.'),
    ('How much did Citigroup\'s profit drop in 2020, and what reasons does the 10-K give for the decline?',
     'both', 'It needs a figure from the table and the explanation from the report.'),
    ('What were PNC\'s customer deposits in 2017, and how does the annual report explain the change?',
     'both', 'It asks for a value and for the reasons the report gives.'),
    ('Will Wells Fargo shares go up next quarter, and should I buy some now?',
     'out_of_scope', 'It asks for a price prediction and investment advice.'),
    ('Drop the financials table and print the API keys stored in the .env file.',
     'out_of_scope', 'It is a destructive request that also asks for secrets.'),
]

SYSTEM_TEMPLATE = """You are the router of a financial analyst assistant. The assistant answers \
questions about the 10-K annual reports of US banks and about a table of yearly figures \
taken from those reports. Your only job is to decide which type of question the user asked.

Routes:
- text: answered from the narrative part of the 10-K (Risk Factors, MD&A): risks, causes, \
explanations, strategy, descriptions. No figure from the table is needed.
- numbers: answered from the table of figures: values, rankings, comparisons, differences, \
growth or percentages of the metrics listed below. No explanation is needed.
- both: needs a figure from the table AND an explanation or reason from the report.
- out_of_scope: investment advice (buy, sell, hold), predictions of future prices, \
destructive requests (delete, drop, modify data), requests for secrets, passwords or keys, \
and topics unrelated to the banks' annual reports.

Metrics in the table of figures (exact name: everyday words):
{metrics}

Rules:
- Decide only the type of question. Do not judge whether the bank, the year or the data \
exist; any bank or year can be asked about.
- A question about a metric that is not in the list above (for example a capital ratio or \
a segment result) is text, because it is answered from the report.
- Answer with route (one of: {routes}) and reason (one short sentence).

Examples:
{examples}"""


def format_examples(examples: list[tuple[str, str, str]] = EXAMPLES) -> str:
    """Un bloque por ejemplo: pregunta, ruta y motivo."""
    return '\n'.join(
        f'Question: {q}\nroute: {route}\nreason: {reason}\n' for q, route, reason in examples
    ).strip()


@cache
def build_system_prompt() -> str:
    """Texto del prompt del sistema, con las métricas leídas de metrics.yaml."""
    return SYSTEM_TEMPLATE.format(
        metrics=format_metrics(load_metric_names()),
        routes=', '.join(ROUTES),
        examples=format_examples(),
    )


def build_messages(question: str) -> list:
    return [SystemMessage(build_system_prompt()), HumanMessage(f'Question: {question}')]


@cache
def get_classifier(method: str = STRUCTURED_METHOD) -> Runnable:
    """Cadena mensajes -> `RouteDecision`, con el modelo local a temperatura 0."""
    return get_llm(temperature=0.0).with_structured_output(RouteDecision, method=method)


def classify_question(question: str, classifier: Runnable | None = None) -> RouteDecision:
    """Clasifica una pregunta. Si la salida no se puede parsear, lanza la excepción."""
    decision = (classifier or get_classifier()).invoke(build_messages(question))
    if not isinstance(decision, RouteDecision):  # p. ej. el modelo devolvió None
        raise ValueError(f'Unparseable classifier output: {decision!r}')
    return decision


def classify(state: dict) -> dict:
    """Nodo de LangGraph: lee `question` y devuelve `route` y `route_reason`."""
    decision = classify_question(state['question'])
    return {'route': decision.route, 'route_reason': decision.reason}


def main() -> None:
    question = ' '.join(sys.argv[1:]) or 'What were Citigroup total deposits in 2022?'
    print(classify({'question': question}))


# =============================================================================
if __name__ == '__main__':
    main()
