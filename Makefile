PYTHON ?= .venv/bin/python

.PHONY: data test

## Reconstruye todos los datos (descargas, DuckDB y manifiesto)
data:
	$(PYTHON) -m financial_analyst_agent.ingest.run_all

test:
	$(PYTHON) -m pytest
