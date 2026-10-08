"""Pruebas de download_10k sin llamadas reales a la SEC (la red se reemplaza por dobles)."""

import pandas as pd
import pytest
import requests

from financial_analyst_agent.ingest import download_10k as d

BANK = {"ticker": "JPM", "name": "JPMorgan Chase", "cik": "0000019617"}

SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K", "8-K", "10-K", "10-K/A", "10-K", "10-K"],
            "accessionNumber": [
                "0000019617-25-000400",
                "0000019617-25-000270",
                "0000019617-25-000100",
                "0000019617-24-000225",
                "0000019617-24-000300",
                "0000019617-23-000231",
                "0000019617-22-000272",
            ],
            "primaryDocument": [
                "q3.htm",
                "jpm-20241231.htm",
                "8k.htm",
                "jpm-20231231.htm",
                "amend.htm",
                "jpm-20221231.htm",
                "jpm-20211231.htm",
            ],
            "filingDate": [
                "2025-11-01",
                "2025-02-14",
                "2025-01-15",
                "2024-02-16",
                "2024-05-01",
                "2023-02-21",
                "2022-02-22",
            ],
            "reportDate": ["x"] * 7,
        }
    }
}


class FakeResponse:
    """Imita una respuesta de requests con JSON, contenido y código de estado."""
    def __init__(self, json_data=None, content=b"", status_code=200):
        self._json = json_data
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")


class FakeSession:
    """Registra las URLs pedidas y responde según el tipo de URL."""

    def __init__(self, submissions=SUBMISSIONS, status_code=200):
        self.submissions = submissions
        self.status_code = status_code
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url.startswith("https://data.sec.gov/submissions/"):
            return FakeResponse(json_data=self.submissions, status_code=self.status_code)
        return FakeResponse(content=f"<html>{url}</html>".encode())


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Reemplaza time.sleep para no esperar y registrar las pausas pedidas."""
    sleeps = []
    monkeypatch.setattr(d.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    """Redirige data/processed y data/raw/10k a una carpeta temporal."""
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw" / "10k"
    monkeypatch.setattr(d, "PROCESSED_DIR", processed)
    monkeypatch.setattr(d, "RAW_10K_DIR", raw)
    return processed, raw


def test_to_ascii_quita_tildes():
    """to_ascii elimina tildes y deja el resto del texto igual."""
    assert d.to_ascii("Richard Andrés correo@x.com") == "Richard Andres correo@x.com"


def test_build_session_usa_user_agent_ascii(tmp_path, monkeypatch):
    """La sesión toma SEC_USER_AGENT del .env y lo envía convertido a ASCII."""
    env = tmp_path / ".env"
    env.write_text("SEC_USER_AGENT=José Núñez jose@x.com\n", encoding="utf-8")
    monkeypatch.setattr(d, "ENV_FILE", env)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    session = d.build_session()

    assert session.headers["User-Agent"] == "Jose Nunez jose@x.com"


def test_build_session_sin_user_agent_termina(tmp_path, monkeypatch):
    """Si SEC_USER_AGENT está vacío, el script se detiene con SystemExit."""
    env = tmp_path / ".env"
    env.write_text("SEC_USER_AGENT=\n", encoding="utf-8")
    monkeypatch.setattr(d, "ENV_FILE", env)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(SystemExit):
        d.build_session()


def test_fetch_recent_filings_usa_cik_de_10_digitos_y_alinea_listas():
    """Completa el CIK a 10 dígitos y convierte las listas paralelas en filas alineadas."""
    session = FakeSession()

    filings = d.fetch_recent_filings(session, "19617")

    assert session.calls == ["https://data.sec.gov/submissions/CIK0000019617.json"]
    assert list(filings.columns) == ["form", "accessionNumber", "primaryDocument", "filingDate"]
    assert len(filings) == 7
    row = filings.iloc[1]
    assert (row.form, row.accessionNumber, row.primaryDocument, row.filingDate) == (
        "10-K",
        "0000019617-25-000270",
        "jpm-20241231.htm",
        "2025-02-14",
    )


def test_fetch_recent_filings_propaga_error_http():
    """Un error HTTP de la SEC (p. ej. 403) se propaga como requests.HTTPError."""
    with pytest.raises(requests.HTTPError):
        d.fetch_recent_filings(FakeSession(status_code=403), BANK["cik"])


def test_latest_10k_filtra_y_toma_los_3_mas_recientes():
    """Conserva solo los 3 10-K más recientes (sin 10-K/A) y agrega los datos del banco."""
    filings = d.fetch_recent_filings(FakeSession(), BANK["cik"])

    selected = d.latest_10k(filings, BANK)

    assert list(selected["form"]) == ["10-K"] * 3
    assert list(selected["filingDate"]) == ["2025-02-14", "2024-02-16", "2023-02-21"]
    assert list(selected.columns[:3]) == ["ticker", "name", "cik"]
    assert set(selected["ticker"]) == {"JPM"}


def test_latest_10k_ordena_aunque_la_respuesta_venga_desordenada():
    """Ordena por filingDate aunque la SEC no entregue los informes en orden."""
    filings = pd.DataFrame(
        {
            "form": ["10-K", "10-K", "10-K", "10-K"],
            "accessionNumber": ["a", "b", "c", "d"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
            "filingDate": ["2021-02-01", "2024-02-01", "2022-02-01", "2023-02-01"],
        }
    )

    selected = d.latest_10k(filings, BANK)

    assert list(selected["accessionNumber"]) == ["b", "d", "c"]


def test_build_document_url_quita_ceros_y_guiones():
    """La URL usa el CIK sin ceros a la izquierda y el accession sin guiones."""
    url = d.build_document_url("0000019617", "0000019617-25-000270", "jpm-20241231.htm")

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/19617/000001961725000270/jpm-20241231.htm"
    )


def test_download_filings_guarda_archivos_y_espera_entre_peticiones(tmp_dirs, no_sleep):
    """Guarda cada documento con prefijo ticker_fecha y espera 0,3 s después de cada descarga."""
    _, raw = tmp_dirs
    table = pd.DataFrame(
        {
            "ticker": ["JPM", "JPM"],
            "filingDate": ["2025-02-14", "2024-02-16"],
            "primaryDocument": ["jpm-20241231.htm", "jpm-20231231.htm"],
            "url": ["https://www.sec.gov/a", "https://www.sec.gov/b"],
        }
    )
    session = FakeSession()

    d.download_filings(session, table)

    assert session.calls == ["https://www.sec.gov/a", "https://www.sec.gov/b"]
    assert (raw / "JPM_2025-02-14_jpm-20241231.htm").read_bytes() == b"<html>https://www.sec.gov/a</html>"
    assert (raw / "JPM_2024-02-16_jpm-20231231.htm").exists()
    assert no_sleep == [0.3, 0.3]


def test_download_filings_omite_archivos_existentes(tmp_dirs, no_sleep):
    """No vuelve a descargar ni sobrescribe un archivo que ya existe."""
    _, raw = tmp_dirs
    raw.mkdir(parents=True)
    existing = raw / "JPM_2025-02-14_jpm-20241231.htm"
    existing.write_text("previo")
    table = pd.DataFrame(
        {
            "ticker": ["JPM"],
            "filingDate": ["2025-02-14"],
            "primaryDocument": ["jpm-20241231.htm"],
            "url": ["https://www.sec.gov/a"],
        }
    )
    session = FakeSession()

    d.download_filings(session, table)

    assert session.calls == []
    assert existing.read_text() == "previo"
    assert no_sleep == []


def test_main_genera_csv_y_descarga_documentos(tmp_dirs, monkeypatch, no_sleep):
    """Flujo completo con 2 bancos: CSV con 3 10-K por banco, URLs correctas y 6 descargas."""
    processed, raw = tmp_dirs
    banks = [BANK, {"ticker": "BAC", "name": "Bank of America", "cik": "0000070858"}]
    session = FakeSession()
    monkeypatch.setattr(d, "build_session", lambda: session)
    monkeypatch.setattr(d, "load_banks", lambda: banks)

    d.main()

    table = pd.read_csv(processed / "filings.csv", dtype=str)
    assert len(table) == 6
    assert list(table.columns) == [
        "ticker", "name", "cik", "form", "accessionNumber",
        "primaryDocument", "filingDate", "url",
    ]
    assert table.groupby("ticker").size().to_dict() == {"BAC": 3, "JPM": 3}
    assert (table["form"] == "10-K").all()
    assert table["cik"].iloc[0] == "0000019617"
    bac_url = table.loc[table["ticker"] == "BAC", "url"].iloc[0]
    assert bac_url.startswith("https://www.sec.gov/Archives/edgar/data/70858/")

    assert len(list(raw.iterdir())) == 6
    # 2 consultas de submissions + 6 descargas, todas con pausa de 0,3 s.
    assert len(session.calls) == 8
    assert no_sleep == [0.3] * 8


def test_load_banks_lee_companies_yaml():
    """Lee los 6 bancos de companies.yaml, todos con CIK de 10 dígitos."""
    banks = d.load_banks()

    assert len(banks) == 6
    assert all(len(bank["cik"]) == 10 for bank in banks)
