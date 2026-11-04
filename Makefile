PYTHON ?= .venv/bin/python

.PHONY: data test

## Reconstruye todos los datos (descargas, texto y secciones de los 10-K, DuckDB y manifiesto)
data:
	$(PYTHON) -m financial_analyst_agent.ingest.run_all

test:
	$(PYTHON) -m pytest
