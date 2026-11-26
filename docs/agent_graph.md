# Agent graph design

Planned LangGraph graph for the Financial Analyst Agent. Only the design is fixed
today; the nodes are built over the next days. Solid arrows are fixed edges, dashed
arrows are conditional edges, and each dashed label is the route name the
classifier returns.

```mermaid
flowchart TD
    start([START])
    classify(classify)
    retrieve_text(retrieve_text)
    query_sql(query_sql)
    answer(answer)
    refuse(refuse)
    finish([END])

    start --> classify
    classify -. text .-> retrieve_text
    classify -. both .-> retrieve_text
    classify -. numbers .-> query_sql
    classify -. out_of_scope .-> refuse
    retrieve_text -. text .-> answer
    retrieve_text -. both .-> query_sql
    query_sql --> answer
    answer --> finish
    refuse --> finish
```

Rendered image (for viewers without Mermaid support): [agent_graph.png](agent_graph.png).
Regenerate it after changing the diagram with
`mmdc -i docs/agent_graph.md -o docs/agent_graph.png` (the output file gets a `-1` suffix; rename it).

Routes: `text`, `numbers`, `both`, `out_of_scope`. The classifier must return these
exact strings. `retrieve_text` has a conditional edge too: it goes to `query_sql`
only when `route == "both"`, otherwise to `answer`.

## Planned state

| Field | Meaning |
|---|---|
| `question` | The user's question, as asked. |
| `route` | One of `text`, `numbers`, `both`, `out_of_scope`, set by `classify`. |
| `route_reason` | One sentence from `classify` on why it chose the route, for debugging and evaluation. |
| `sources` | Cited 10-K chunks (company, year, section) from `retrieve_text`; empty if none. |
| `sql` | The DuckDB query `query_sql` ran (after validation and any repair). |
| `rows` | Rows returned by that query; empty on error or no match. |
| `answer` | Final text written by `answer` or `refuse`. |
| `error` | Tool failure message (RAG or SQL), or `None`; `answer` must report it, not hide it. |

## Node to code mapping

| Node | Code | Status |
|---|---|---|
| `classify` | `src/financial_analyst_agent/agent/classifier.py`: `classify(state)` returns `route` and `route_reason`; `RouteDecision` and `ROUTES` in `src/financial_analyst_agent/agent/schemas.py` | Done (26 Nov, structured output with `json_schema`) |
| `retrieve_text` | `src/financial_analyst_agent/rag/chain.py`: `get_chain(grounded=True)` returns `answer`, `quote`, `sources`; `citations()` turns sources into labels | Exists (RAG chain with citations) |
| `query_sql` | `src/financial_analyst_agent/sql/tool.py`: `answer_with_sql(question)` (also wrapped as the `text_to_sql` tool) returns `sql`, `columns`, `rows`, `error`, `attempts` | Exists (text-to-SQL tool) |
| `answer` | — | Pending, built on 27 Nov |
| `refuse` | — | Pending, built on 30 Nov |

Both existing modules use the local Ollama model through
`src/financial_analyst_agent/llm.py` (`get_llm`).

## Design notes

- **Investment advice is out of scope.** Questions such as "Should I buy JPM stock?"
  are classified as `out_of_scope` and go to `refuse`. The agent reports what the
  10-K filings say; it does not recommend buying or selling.
- **Tool failures reach `answer` with the error.** If `retrieve_text` or `query_sql`
  fails (for example, the SQL is rejected as unsafe or still fails after retries),
  the node writes the message to `error` and the graph continues to `answer`, which
  says the figure could not be retrieved. The agent never fills the gap with made-up
  numbers.
