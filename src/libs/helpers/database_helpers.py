"""Database query helpers to prevent repeating cursor / commit / rollback in code."""

from contextlib import contextmanager
from typing import Iterable

@contextmanager
def transaction(conn):
    """
    Make several helper calls ONE all-or-nothing transaction. Pass commit=False
    to each helper inside the block; this commits once on success and rolls everything
    back if anything raises.

    Parameters
    ----------
        conn: Connection to the database.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_database(conn, stmts: tuple[str, ...]) -> None:
    """
    Initialise the database.

    Parameters
    ----------
        conn:  Connection to the database.
        stmts: SQL strings.
    """
    try:
        with conn.cursor() as cur:
            for stmt in stmts:
                cur.execute(stmt)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def fetch_one(conn, sql: str, params: tuple = (), commit: bool = True) -> list:
    """
    Run a query, return one row (or None). Commits when done unless specified.

    Parameters
    ----------
        conn:   Connection to the database.
        sql:    SQL string.
        params: Parameters in the SQL.

    Returns
    -------
        - Row column.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()

        if commit:
            conn.commit()

        return row

    except Exception:
        conn.rollback()
        raise


def fetch_all(conn, sql: str, params: tuple = ()) -> list:
    """
    Parameters
    ----------
        conn:   Connection to the database.
        sql:    SQL string.
        params: Parameters in the SQL.

    Returns
    -------
        - Row columns.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise


def execute(conn, sql: str, params: tuple = (), commit: bool = True) -> int:
    """
    Run INSERT/UPDATE/DELETE. Returns teh number of rows affected.

    Parameters
    ----------
        conn:   Connection to the database.
        sql:    SQL string.
        params: Parameters in the SQL.
        commit: If commit the execution.

    Returns
    -------
        - Number of executions.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            count = cur.rowcount
        if commit:
            conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise


def execute_many(
    conn, sql: str, params_seq: Iterable[tuple], commit: bool = True
) -> None:
    """
    Runn one INSERT/UPDATE once per params tuple. No-op if params_seq is empty.

    Parameters
    ----------
        conn:       Connection to the database.
        sql:        SQL string.
        params_seq: Parameters sequences in the SQL.
        commit:     If commit the execution.
    """
    rows = list(params_seq)
    if not rows:
        return

    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
