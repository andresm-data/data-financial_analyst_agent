"""Validación del SQL generado antes de ejecutarlo, sobre el árbol de sqlglot.

Todas las comprobaciones miran el árbol sintáctico (dialecto `duckdb`), no el
texto: una cadena `'DROP'` o una columna `last_update` no bloquean nada.

Capas, en este orden:
1. Comandos prohibidos (DROP, DELETE, UPDATE, INSERT, ATTACH, COPY) en
   cualquier parte del árbol, también dentro de CTE y subconsultas.
2. Lectura de archivos desde un SELECT: funciones como `read_csv` o `glob`
   y tablas cuyo nombre es una ruta (`FROM 'datos.csv'`).
3. Una sola sentencia, y que sea un SELECT (con o sin CTE), sin UNION.
4. LIMIT de como máximo `MAX_LIMIT` filas en la consulta principal.

`validate_sql` devuelve el SQL listo para ejecutar o lanza:
- `UnsafeSQLError`: la consulta está prohibida (varias sentencias, algo que no
  es un SELECT, UNION, comandos o funciones prohibidos). No se reintenta.
- `InvalidSQLError`: el texto está vacío o no se puede parsear. Es un error de
  sintaxis como cualquier otro, así que sí se puede pedir una corrección.

Uso: `poetry run python -m financial_analyst_agent.sql.validator "SQL"`.
"""
import sys

import sqlglot
from sqlglot import exp

# =============================================================================
DIALECT = 'duckdb'
MAX_LIMIT = 100

# Tipo de nodo -> por qué se prohíbe. Cada uno tiene su propio mensaje.
FORBIDDEN_COMMANDS: dict[type[exp.Expression], str] = {
    exp.Drop: 'DROP no está permitido: la herramienta no puede borrar tablas ni otros objetos.',
    exp.Delete: 'DELETE no está permitido: la herramienta no puede borrar filas.',
    exp.Update: 'UPDATE no está permitido: la herramienta no puede modificar filas.',
    exp.Insert: 'INSERT no está permitido: la herramienta no puede añadir filas.',
    exp.Attach: 'ATTACH no está permitido: la herramienta no puede abrir otras bases de datos.',
    exp.Copy: 'COPY no está permitido: la herramienta no puede leer ni escribir archivos.',
}

# Funciones de DuckDB que leen archivos del disco o de la red. Además se
# bloquea cualquier otra que empiece por `read_` (read_ndjson_auto, ...).
FILE_FUNCTIONS = {
    'read_csv', 'read_csv_auto', 'read_parquet', 'parquet_scan', 'read_json',
    'read_json_auto', 'read_json_objects', 'read_ndjson', 'read_text',
    'read_blob', 'read_xlsx', 'sniff_csv', 'glob',
}
PATH_CHARS = ('/', '\\', '.')


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


def check_forbidden_commands(tree: exp.Expression) -> None:
    """Rechaza DROP, DELETE, UPDATE, INSERT, ATTACH y COPY en todo el árbol."""
    for node in tree.walk():
        message = FORBIDDEN_COMMANDS.get(type(node))
        if message:
            where = '' if node is tree else ' (aparece dentro de un CTE o subconsulta)'
            raise UnsafeSQLError(message + where)


def function_name(node: exp.Func) -> str:
    """Nombre en minúsculas de una función, tenga o no clase propia en sqlglot."""
    name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
    return name.lower()


def check_no_file_access(tree: exp.Expression) -> None:
    """Rechaza funciones que leen archivos y tablas cuyo nombre es una ruta."""
    for node in tree.find_all(exp.Func):
        name = function_name(node)
        if name in FILE_FUNCTIONS or name.startswith('read_'):
            raise UnsafeSQLError(
                f'La función {name}() no está permitida: lee archivos del disco o de la red. '
                'Consulta solo la tabla financials.'
            )
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier) and any(c in table.name for c in PATH_CHARS):
            raise UnsafeSQLError(
                f'La tabla "{table.name}" parece una ruta de archivo y DuckDB la leería del disco. '
                'Consulta solo la tabla financials.'
            )


def check_select_only(tree: exp.Expression) -> None:
    """Acepta un SELECT (con o sin CTE); rechaza todo lo demás y UNION."""
    if isinstance(tree, exp.SetOperation) or tree.find(exp.SetOperation):
        kind = type(tree.find(exp.SetOperation) or tree).__name__.upper()
        raise UnsafeSQLError(f'{kind} no está permitido; escribe un único SELECT.')
    if not isinstance(tree, exp.Select):
        kind = tree.key.upper()
        raise UnsafeSQLError(f'Solo se permiten consultas SELECT; se recibió {kind}.')


def limit_value(limit: exp.Expression | None) -> int | None:
    """Número de filas de un LIMIT (o FETCH FIRST) literal; si no, None.

    LIMIT con porcentaje, expresiones o negativos devuelven None, así que se
    sustituyen por el máximo.
    """
    if isinstance(limit, exp.Limit):
        options = limit.args.get('limit_options')
        if options is not None and options.args.get('percent'):
            return None
        count = limit.expression
    elif isinstance(limit, exp.Fetch):
        count = limit.args.get('count')
    else:
        return None
    if isinstance(count, exp.Literal) and count.is_int:
        return int(count.name)
    return None


def enforce_limit(tree: exp.Select, max_rows: int = MAX_LIMIT) -> exp.Select:
    """Sin LIMIT -> LIMIT max_rows; si es mayor, baja a max_rows; si no, igual."""
    current = limit_value(tree.args.get('limit'))
    if current is None or current > max_rows:
        tree = tree.limit(max_rows, copy=False)
    return tree


def validate_sql(sql: str) -> str:
    """Devuelve el SQL validado y con LIMIT, regenerado por sqlglot (duckdb)."""
    tree = parse_single_statement(sql)
    check_forbidden_commands(tree)
    check_no_file_access(tree)
    check_select_only(tree)
    return enforce_limit(tree).sql(dialect=DIALECT, pretty=True)


# =============================================================================
if __name__ == '__main__':
    try:
        print(validate_sql(' '.join(sys.argv[1:])))
    except SQLValidationError as exc:
        print(f'{type(exc).__name__}: {exc}')
        sys.exit(1)
