"""Evalúa la cadena RAG con RAGAS: faithfulness y answer relevancy.

Usa las preguntas `type = text` de eval/questions.csv y el índice de Chroma
que indique config/rag.yaml, así que para comparar tamaños de chunk solo se
cambia la config. Todo corre en local con Ollama:

- Generador: la cadena RAG de siempre (`chain.get_chain`, con cita, búsqueda
  híbrida + rerank, k=5, llama3.2:3b).
- Juez: `JUDGE_MODEL` a temperatura `JUDGE_TEMPERATURE`, de otra familia que
  el generador para que no se califique a sí mismo.
- Embeddings de answer relevancy: bge-small, los mismos del índice.

Dos fases, para no tener los dos modelos en memoria a la vez y poder retomar:
1. La cadena responde cada pregunta; respuesta y contextos se guardan en
   data/eval/ (ignorada por git). Las ya respondidas no se repiten.
2. RAGAS califica todas las respuestas y escribe eval/results/chunk_<size>.csv
   con un puntaje por pregunta y la configuración en cada fila.

RAGAS califica "I do not know" con 0 en las dos métricas; la columna
`no_answer` lo marca para poder separar abstenciones de respuestas malas.

Uso: `poetry run ragas_eval` o `poetry run ragas_eval --limit 2` (prueba corta).
"""
import argparse
import json
import math
import os
import time
from datetime import date
from pathlib import Path

# RAGAS manda estadísticas de uso a sus servidores si no se desactiva.
os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')

import pandas as pd  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402

from financial_analyst_agent.embeddings import MODEL_NAME as EMBEDDINGS_MODEL  # noqa: E402
from financial_analyst_agent.ingest.download_10k import PROJECT_ROOT  # noqa: E402
from financial_analyst_agent.llm import MODEL_NAME as GENERATOR_MODEL  # noqa: E402
from financial_analyst_agent.rag.chain import is_no_answer  # noqa: E402
from financial_analyst_agent.rag.chunking import load_chunking_config, load_index_paths  # noqa: E402
from financial_analyst_agent.rag.retrieval import K  # noqa: E402

# =============================================================================
QUESTIONS_FILE = PROJECT_ROOT / 'eval' / 'questions.csv'
RESULTS_DIR = PROJECT_ROOT / 'eval' / 'results'
ANSWERS_DIR = PROJECT_ROOT / 'data' / 'eval'

JUDGE_MODEL = 'qwen2.5:3b'
JUDGE_TEMPERATURE = 0.0
# Contexto del juez en tokens: cabe el prompt de faithfulness con 5 chunks de 1000.
JUDGE_NUM_CTX = 8192
# Segundos por llamada al juez; en CPU un prompt largo tarda más de un minuto.
JUDGE_TIMEOUT = 900

METRICS = ['faithfulness', 'answer_relevancy']
RESULT_COLUMNS = [
    'id', 'question', 'answer', 'no_answer', 'n_contexts', *METRICS, 'error',
    'chunk_size', 'chunk_overlap', 'k', 'chroma_dir',
    'generator_model', 'judge_model', 'judge_temperature', 'embeddings_model', 'run_date',
]


# =============================================================================
def load_text_questions(path: Path = QUESTIONS_FILE) -> pd.DataFrame:
    """Preguntas `type = text` de eval/questions.csv, en orden de `id`."""
    questions = pd.read_csv(path)
    return questions[questions['type'] == 'text'].sort_values('id').reset_index(drop=True)


# =============================================================================
def answer_question(chain: Runnable, question: str) -> dict:
    """Respuesta y contextos de una pregunta; si la cadena falla, el error."""
    try:
        result = chain.invoke(question)
    except Exception as exc:  # se registra y se sigue: no se descarta en silencio
        return {'answer': '', 'contexts': [], 'error': f'{type(exc).__name__}: {exc}'}

    return {
        'answer': result['answer'],
        'contexts': [doc.page_content for doc in result['context']],
        'error': '',
    }


def load_answers(path: Path) -> dict[int, dict]:
    """Respuestas ya guardadas por `answer_all`, por `id` de pregunta."""
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as f:
        return {row['id']: row for row in map(json.loads, f)}


def answer_all(chain_factory, questions: pd.DataFrame, path: Path) -> list[dict]:
    """Responde las preguntas que falten en `path` y devuelve todas en orden.

    Las que fallaron antes se vuelven a intentar. La cadena se crea solo si
    hay algo que responder.
    """
    saved = {id_: row for id_, row in load_answers(path).items() if not row['error']}
    pending = questions[~questions['id'].isin(saved)]
    if len(pending):
        chain = chain_factory()
        path.parent.mkdir(parents=True, exist_ok=True)
        for n, row in enumerate(pending.itertuples(), 1):
            start = time.perf_counter()
            saved[row.id] = {'id': int(row.id), 'question': row.question,
                             **answer_question(chain, row.question)}
            print(f'[{n}/{len(pending)}] id={row.id} {time.perf_counter() - start:.0f} s '
                  f'{saved[row.id]["error"] or saved[row.id]["answer"][:80]!r}')
            # Se reescribe el archivo entero para no dejar intentos fallidos repetidos.
            with path.open('w', encoding='utf-8') as f:
                for item in sorted(saved.values(), key=lambda r: r['id']):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return [saved[id_] for id_ in questions['id'] if id_ in saved]


# =============================================================================
def get_judge():
    """Juez de RAGAS: `JUDGE_MODEL` en Ollama, en modo JSON."""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    from financial_analyst_agent.llm import DEFAULT_BASE_URL

    return LangchainLLMWrapper(ChatOllama(
        model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE, num_ctx=JUDGE_NUM_CTX,
        format='json', base_url=os.getenv('OLLAMA_BASE_URL') or DEFAULT_BASE_URL,
    ))


def score(answers: list[dict]) -> pd.DataFrame:
    """Faithfulness y answer relevancy de cada respuesta, en el mismo orden.

    Las preguntas que fallaron en la cadena no se mandan al juez y quedan con
    NaN; las que el juez no pudo calificar también quedan con NaN.
    """
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    from financial_analyst_agent.embeddings import get_embeddings

    scores = pd.DataFrame(math.nan, index=range(len(answers)), columns=METRICS)
    ok = [i for i, row in enumerate(answers) if not row['error']]
    if not ok:
        return scores

    dataset = EvaluationDataset.from_list([
        {'user_input': answers[i]['question'], 'response': answers[i]['answer'],
         'retrieved_contexts': answers[i]['contexts']}
        for i in ok
    ])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=get_judge(),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        # Una llamada a la vez: en CPU, varias en paralelo solo se estorban.
        run_config=RunConfig(timeout=JUDGE_TIMEOUT, max_retries=3, max_workers=1),
    ).to_pandas()
    scores.loc[ok, METRICS] = result[METRICS].to_numpy()

    return scores


# =============================================================================
def build_results(answers: list[dict], scores: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Una fila por pregunta con sus puntajes y la configuración del experimento."""
    config = load_chunking_config()
    results = pd.DataFrame({
        'id': [row['id'] for row in answers],
        'question': [row['question'] for row in answers],
        'answer': [row['answer'] for row in answers],
        'no_answer': [not row['error'] and is_no_answer(row['answer']) for row in answers],
        'n_contexts': [len(row['contexts']) for row in answers],
        **{metric: scores[metric].to_numpy() for metric in METRICS},
        'error': [row['error'] for row in answers],
    })
    results = results.assign(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        k=K,
        chroma_dir=load_index_paths().chroma_dir.relative_to(PROJECT_ROOT).as_posix(),
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL,
        judge_temperature=JUDGE_TEMPERATURE,
        embeddings_model=EMBEDDINGS_MODEL,
        run_date=run_date,
    )
    return results[RESULT_COLUMNS]


def summary(results: pd.DataFrame) -> str:
    """Promedio de cada métrica, cuántas preguntas tienen NaN y cuáles, y abstenciones."""
    lines = [f"I do not know: {int(results['no_answer'].sum())}/{len(results)}"]
    for metric in METRICS:
        missing = results.loc[results[metric].isna(), 'id'].tolist()
        lines.append(
            f'{metric}: {results[metric].mean():.3f} '
            f'({results[metric].notna().sum()}/{len(results)} calificadas; NaN en id={missing})'
        )
    return '\n'.join(lines)


# =============================================================================
def main(argv: list[str] | None = None) -> None:
    """Responde, califica y guarda eval/results/chunk_<size>.csv."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--limit', type=int, help='solo las primeras N preguntas (prueba corta)')
    parser.add_argument('--output', type=Path, help='CSV de salida (por defecto chunk_<size>.csv)')
    args = parser.parse_args(argv)

    chunk_size = load_chunking_config().chunk_size
    chroma_dir = load_index_paths().chroma_dir
    questions = load_text_questions()
    if args.limit:
        questions = questions.head(args.limit)
    print(f'{len(questions)} preguntas | chunk_size={chunk_size} | {chroma_dir} | juez {JUDGE_MODEL}')

    def chain_factory():
        from financial_analyst_agent.rag.chain import get_chain
        return get_chain(grounded=True)

    answers = answer_all(chain_factory, questions, ANSWERS_DIR / f'ragas_chunk_{chunk_size}_answers.jsonl')

    start = time.perf_counter()
    results = build_results(answers, score(answers), date.today().isoformat())
    print(f'Juez: {time.perf_counter() - start:.0f} s')

    output = args.output or RESULTS_DIR / f'chunk_{chunk_size}.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(summary(results))
    print(f'Guardado en {output}')


# =============================================================================
if __name__ == '__main__':
    main()
