"""Prueba el clasificador del agente con 10 preguntas de eval/questions.csv.

Regla de selección (fijada antes de ver resultados, no se cambia después):
  - ruta esperada desde `type`: text -> text, figure -> numbers (no a mano);
  - las filas `control` (ids 41-45) quedan fuera, como reserva del SQL;
  - rng = random.Random(42); 5 ids con rng.sample() de los `text` ordenados por
    id y luego 5 con rng.sample() de los `figure` ordenados por id.

Cada pregunta va en un try/except: si la salida no se puede parsear, se guarda
el error, cuenta como incorrecta y se sigue (sin lógica de respaldo).

Aparte, sin contar para el 8 de 10, corre una mini-lista de 6 preguntas
inventadas (2 both, 2 out_of_scope, 2 casos límite text/numbers) que el CSV no
cubre. No se añaden a eval/questions.csv.

Uso: `poetry run python eval/run_classifier_eval.py [run]`  (por defecto run1)
Salida: eval/results/classifier_<run>.csv, classifier_extra_check.csv y
classifier_<run>_meta.json (modelo, temperatura, fecha, commit, regla).
"""
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from financial_analyst_agent.agent.classifier import STRUCTURED_METHOD, classify
from financial_analyst_agent.llm import MODEL_NAME

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_CSV = EVAL_DIR / 'questions.csv'
RESULTS_DIR = EVAL_DIR / 'results'

SEED = 42
PER_TYPE = 5
TYPE_TO_ROUTE = {'text': 'text', 'figure': 'numbers'}
SELECTION_RULE = (
    "expected_route from type (text->text, figure->numbers); control rows excluded; "
    "rng=random.Random(42); rng.sample(text ids sorted, 5) then rng.sample(figure ids sorted, 5)"
)
TEMPERATURE = 0.0

# Mini-lista aparte: (id, caso, pregunta, ruta esperada). Inventadas.
EXTRA_QUESTIONS = [
    ('x1', 'both', "How large were Goldman Sachs' total liabilities in 2023, and what does the report say drove the change?", 'both'),
    ('x2', 'both', 'Which bank grew its deposits the most from 2019 to 2021, and how does its 10-K explain that growth?', 'both'),
    ('x3', 'out_of_scope', 'Is Truist a better investment than Bank of America for my retirement savings?', 'out_of_scope'),
    ('x4', 'out_of_scope', "What will Morgan Stanley's share price be at the end of 2027?", 'out_of_scope'),
    ('x5', 'edge: metric not in table', "What was JPMorgan Chase's CET1 capital ratio in 2024?", 'text'),
    ('x6', 'edge: figures asked narratively', "How did Citigroup's total deposits change between 2021 and 2023?", 'numbers'),
]


def select_questions(questions: pd.DataFrame) -> pd.DataFrame:
    """Aplica la regla fija de selección; devuelve id, question, expected_route."""
    rng = random.Random(SEED)
    ids = []
    for type_ in TYPE_TO_ROUTE:
        pool = sorted(questions.loc[questions['type'] == type_, 'id'])
        ids += rng.sample(pool, PER_TYPE)
    chosen = questions.set_index('id').loc[ids].reset_index()
    chosen['expected_route'] = chosen['type'].map(TYPE_TO_ROUTE)
    return chosen[['id', 'question', 'expected_route']]


def run_one(question: str) -> dict:
    try:
        out = classify({'question': question})
        return {'real_route': out['route'], 'reason': out['route_reason'], 'error': ''}
    except Exception as exc:  # salida no parseable u otro fallo: cuenta como error
        return {'real_route': 'ERROR', 'reason': '', 'error': f'{type(exc).__name__}: {exc}'}


def run(rows: pd.DataFrame) -> pd.DataFrame:
    results = []
    for row in rows.to_dict('records'):
        result = {**row, **run_one(row['question'])}
        result['correct'] = result['real_route'] == row['expected_route']
        print(f"[{'OK ' if result['correct'] else 'BAD'}] {row['id']}: "
              f"{row['expected_route']} -> {result['real_route']}", flush=True)
        results.append(result)
    return pd.DataFrame(results)


def confusion(results: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(results['expected_route'], results['real_route'],
                       rownames=['expected'], colnames=['real'])


def git(*args: str) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True,
                          cwd=EVAL_DIR.parent).stdout.strip()


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'run1'
    columns = ['id', 'question', 'expected_route', 'real_route', 'reason', 'correct', 'error']

    chosen = select_questions(pd.read_csv(QUESTIONS_CSV))
    print('Selected ids:', chosen['id'].tolist())
    main_results = run(chosen)[columns]

    extra = pd.DataFrame(EXTRA_QUESTIONS, columns=['id', 'case', 'question', 'expected_route'])
    extra_results = run(extra)[['id', 'case', *columns[1:]]]

    RESULTS_DIR.mkdir(exist_ok=True)
    main_results.to_csv(RESULTS_DIR / f'classifier_{run_name}.csv', index=False)
    extra_results.to_csv(RESULTS_DIR / 'classifier_extra_check.csv', index=False)
    meta = {
        'run': run_name,
        'date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': MODEL_NAME,
        'temperature': TEMPERATURE,
        'structured_output_method': STRUCTURED_METHOD,
        'code_commit': git('rev-parse', '--short', 'HEAD'),
        'uncommitted_files': git('status', '--porcelain').splitlines(),
        'selection_rule': SELECTION_RULE,
        'selected_ids': chosen['id'].tolist(),
        'correct': int(main_results['correct'].sum()),
        'total': len(main_results),
        'extra_correct': int(extra_results['correct'].sum()),
        'extra_total': len(extra_results),
    }
    (RESULTS_DIR / f'classifier_{run_name}_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8')

    print(f"\nMain: {meta['correct']}/{meta['total']} correct")
    print(confusion(main_results))
    print(f"\nExtra check (not counted): {meta['extra_correct']}/{meta['extra_total']} correct")
    print(confusion(extra_results))


if __name__ == '__main__':
    main()
