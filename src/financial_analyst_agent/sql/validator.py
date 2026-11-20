"""Validación del SQL generado antes de ejecutarlo, sobre el árbol de sqlglot.

Todas las comprobaciones miran el árbol sintáctico (dialecto `duckdb`), no el
texto: una cadena `'DROP'` o una columna `last_update` no bloquean nada.

`validate_sql` devuelve el SQL listo para ejecutar o lanza:
- `UnsafeSQLError`: la consulta está prohibida (varias sentencias, algo que no
  es un SELECT, UNION). No se reintenta.
- `InvalidSQLError`: el texto está vacío o no se puede parsear. Es un error de
  sintaxis como cualquier otro, así que sí se puede pedir una corrección.

Uso: `poetry run python -m financial_analyst_agent.sql.validator "SQL"`.
"""
import sys

import sqlglot
from sqlglot import exp

# =============================================================================
DIALECT = 'duckdb'


# =============================================================================
class SQLValidationError(ValueError):
    """Base de los rechazos del validador."""


class UnsafeSQLError(SQLValidationError):
    """La consulta se rechaza por seguridad; no se debe reintentar."""


class InvalidSQLError(SQLValidationError):
    """El texto está vacío o no es SQL que sqlglot pueda parsear."""


# =============================================================================
def parse_single_statement(sql: str) -> exp.Expression:
    """Parsea el texto y devuelve su única sentencia.

    Las sentencias vacías (por ejemplo, la que deja un `;` final) se ignoran.
    """
    if not sql or not sql.strip():
        raise InvalidSQLError('La consulta está vacía: el modelo no devolvió SQL.')
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except sqlglot.errors.ParseError as exc:
        first_line = str(exc).splitlines()[0]
        raise InvalidSQLError(f'No se pudo parsear el SQL: {first_line}') from exc

    if not statements:
        raise InvalidSQLError('La consulta está vacía: solo contiene ";" o comentarios.')
    if len(statements) > 1:
        raise UnsafeSQLError(
            f'Se esperaba una sola sentencia y hay {len(statements)}; '
            'solo se permite un SELECT.'
        )
    return statements[0]


def check_select_only(tree: exp.Expression) -> None:
    """Acepta un SELECT (con o sin CTE); rechaza todo lo demás y UNION."""
    if isinstance(tree, exp.SetOperation) or tree.find(exp.SetOperation):
        kind = type(tree.find(exp.SetOperation) or tree).__name__.upper()
        raise UnsafeSQLError(f'{kind} no está permitido; escribe un único SELECT.')
    if not isinstance(tree, exp.Select):
        kind = tree.key.upper()
        raise UnsafeSQLError(f'Solo se permiten consultas SELECT; se recibió {kind}.')


def validate_sql(sql: str) -> str:
    """Devuelve el SQL validado, regenerado por sqlglot en dialecto duckdb."""
    tree = parse_single_statement(sql)
    check_select_only(tree)
    return tree.sql(dialect=DIALECT, pretty=True)


# =============================================================================
if __name__ == '__main__':
    try:
        print(validate_sql(' '.join(sys.argv[1:])))
    except SQLValidationError as exc:
        print(f'{type(exc).__name__}: {exc}')
        sys.exit(1)
