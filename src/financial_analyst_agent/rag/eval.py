"""Evalúa el RAG con las preguntas de config/eval_questions.yaml.

Dos niveles:
- Recuperación (segundos, sin LLM): hit@k y MRR, es decir, si algún chunk
  esperado llega al top k y en qué puesto.
- Respuesta (minutos, con llama3.2:3b): cuántas respuestas son correctas,
  cuántas dicen "I do not know" cuando sí había respuesta (missed), cuántas
  responden mal (wrong) y cuántas inventan algo que no está en los 10-K.

Uso: `poetry run rag_eval` o `poetry run rag_eval --retrieval-only`. Cada
ejecución se guarda en data/eval/ (ignorada por git) para comparar cambios.
"""
import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from langchain_core.runnables import Runnable

from financial_analyst_agent.ingest.download_10k import PACKAGE_DIR, PROJECT_ROOT
from financial_analyst_agent.llm import get_llm
from financial_analyst_agent.rag.chain import build_chain, build_grounded_chain, default_retriever, is_no_answer
from financial_analyst_agent.rag.vectorstore import get_vectorstore

# =============================================================================
EVAL_FILE = PACKAGE_DIR / 'config' / 'eval_questions.yaml'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'eval'


# =============================================================================
@dataclass(frozen=True)
class EvalQuestion:
    """Pregunta de evaluación con lo que se espera recuperar y responder."""

    text: str
    expected_chunks: list[str] = field(default_factory=list)
    expected_answer: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        """True si la respuesta está en los 10-K."""
        return bool(self.expected_answer)


def load_questions(path: Path = EVAL_FILE) -> list[EvalQuestion]:
    """Lee las preguntas de evaluación del YAML."""
    with open(path, encoding='utf-8') as f:
        return [EvalQuestion(**item) for item in yaml.safe_load(f)['questions']]


# =============================================================================
def first_hit_rank(retrieved_ids: list[str], expected: list[str]) -> int | None:
    """Puesto (desde 1) del primer chunk esperado, o None si no llegó."""
    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in expected:
            return rank
    return None


def retrieval_summary(ranks: list[int | None]) -> dict:
    """hit@k y MRR a partir del puesto del primer acierto de cada pregunta."""
    n = len(ranks)
    return {
        'questions': n,
        'hit_at_k': sum(rank is not None for rank in ranks) / n if n else 0.0,
        'mrr': sum(1 / rank for rank in ranks if rank) / n if n else 0.0,
    }


def grade_answer(question: EvalQuestion, answer: str) -> str:
    """Califica una respuesta: correct, missed, wrong o invented."""
    if not question.answerable:
        return 'correct' if is_no_answer(answer) else 'invented'
    if is_no_answer(answer):
        return 'missed'

    text = answer.lower()
    found = any(expected.lower() in text for expected in question.expected_answer)
    bad = any(word.lower() in text for word in question.forbidden)

    return 'correct' if found and not bad else 'wrong'


# =============================================================================
def evaluate_retrieval(retriever: Runnable, questions: list[EvalQuestion]) -> dict:
    """Recupera para cada pregunta con respuesta y mide hit@k y MRR."""
    rows = []
    for question in questions:
        if not question.expected_chunks:
            continue
        ids = [doc.metadata['chunk_id'] for doc in retriever.invoke(question.text)]
        rows.append({'question': question.text, 'rank': first_hit_rank(ids, question.expected_chunks),
                     'retrieved': ids})

    return {'summary': retrieval_summary([row['rank'] for row in rows]), 'rows': rows}


def evaluate_answers(chain: Runnable, questions: list[EvalQuestion]) -> dict:
    """Ejecuta la cadena con cada pregunta y califica las respuestas."""
    rows = []
    for question in questions:
        start = time.perf_counter()
        result = chain.invoke(question.text)
        rows.append({
            'question': question.text,
            'answer': result['answer'],
            'grade': grade_answer(question, result['answer']),
            'quote': result.get('quote', ''),
            'sources': [doc.metadata['chunk_id'] for doc in result.get('sources', [])],
            'seconds': round(time.perf_counter() - start, 1),
        })

    grades = Counter(row['grade'] for row in rows)
    summary = {grade: grades.get(grade, 0) for grade in ('correct', 'missed', 'wrong', 'invented')}
    summary['questions'] = len(rows)

    return {'summary': summary, 'rows': rows}


# =============================================================================
def print_retrieval(report: dict) -> None:
    """Imprime el puesto del primer acierto de cada pregunta y el resumen."""
    print('\nRecuperación (puesto del primer chunk esperado; - = no llegó)')
    for row in report['rows']:
        print(f"  {row['rank'] or '-':>2}  {row['question']}")
    s = report['summary']
    print(f"  hit@k={s['hit_at_k']:.2f}  MRR={s['mrr']:.2f}  ({s['questions']} preguntas)")


def print_answers(report: dict) -> None:
    """Imprime la calificación de cada respuesta y el resumen."""
    print('\nRespuestas')
    for row in report['rows']:
        answer = ' '.join(row['answer'].split())[:110]
        print(f"  {row['grade']:<8} {row['seconds']:>5}s  {row['question']}\n           -> {answer}")
    s = report['summary']
    print(f"  correct={s['correct']} missed={s['missed']} wrong={s['wrong']} "
          f"invented={s['invented']}  ({s['questions']} preguntas)")


def save_report(report: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Guarda el informe en data/eval/ con la fecha y hora (UTC) en el nombre."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


# =============================================================================
def main() -> None:
    """Evalúa la recuperación y, salvo --retrieval-only, las respuestas."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--retrieval-only', action='store_true', help='no llama al LLM')
    parser.add_argument('--plain', action='store_true', help='cadena sin cita verificada')
    args = parser.parse_args()

    questions = load_questions()
    retriever = default_retriever(get_vectorstore())
    report = {
        'questions_file': str(EVAL_FILE.relative_to(PROJECT_ROOT)),
        'chain': 'plain' if args.plain else 'grounded',
        'retrieval': evaluate_retrieval(retriever, questions),
    }
    print_retrieval(report['retrieval'])

    if not args.retrieval_only:
        chain = (build_chain(retriever, get_llm()) if args.plain
                 else build_grounded_chain(retriever, get_llm(json_mode=True)))
        report['answers'] = evaluate_answers(chain, questions)
        print_answers(report['answers'])

    print(f'\nInforme guardado en {save_report(report)}')


# =============================================================================
if __name__ == '__main__':
    main()
