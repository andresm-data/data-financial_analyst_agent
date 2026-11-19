"""Pruebas del diccionario de nombres de métricas (sin la base de datos)."""
import yaml

from financial_analyst_agent.ingest.download_10k import PACKAGE_DIR
from financial_analyst_agent.ingest.load_financials import METRIC_TAGS

METRICS_FILE = PACKAGE_DIR / 'config' / 'metrics.yaml'


def load():
    return yaml.safe_load(METRICS_FILE.read_text(encoding='utf-8'))['metrics']


def test_targets_are_exactly_the_six_loaded_metrics():
    assert set(load()) == set(METRIC_TAGS)


def test_synonyms_are_lowercase_unique_and_several_per_metric():
    metrics = load()
    synonyms = [s for words in metrics.values() for s in words]

    assert all(len(words) >= 2 for words in metrics.values())
    assert all(s == s.lower().strip() for s in synonyms)
    assert len(synonyms) == len(set(synonyms))  # ningún sinónimo apunta a dos métricas
