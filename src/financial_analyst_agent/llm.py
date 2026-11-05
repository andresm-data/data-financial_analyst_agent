"""Modelo de lenguaje local servido por Ollama, usado desde LangChain.

Requiere el servidor de Ollama en marcha y el modelo descargado:

    ollama pull llama3.2:3b

La URL del servidor se lee de OLLAMA_BASE_URL en el `.env` del paquete; si no
está definida se usa la de Ollama por defecto (http://localhost:11434).
"""
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from financial_analyst_agent.ingest.download_10k import ENV_FILE

MODEL_NAME = 'llama3.2:3b'
DEFAULT_BASE_URL = 'http://localhost:11434'


def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Devuelve el chat de llama3.2:3b conectado al servidor local de Ollama."""
    load_dotenv(ENV_FILE)
    base_url = os.getenv('OLLAMA_BASE_URL') or DEFAULT_BASE_URL

    return ChatOllama(model=MODEL_NAME, base_url=base_url, temperature=temperature)


def main() -> None:
    """Hace una pregunta de prueba al modelo local e imprime la respuesta."""
    question = 'In one sentence, what is a 10-K annual report?'
    answer = get_llm().invoke(question)

    print(f'Pregunta: {question}')
    print(f'Respuesta ({MODEL_NAME}): {answer.content}')


# =============================================================================
if __name__ == '__main__':
    main()
