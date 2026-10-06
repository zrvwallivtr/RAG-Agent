"""Database query helpers to prevent repeating cursor / commit / rollback in code."""


def init_database(conn, stmts: tuple[str, ...]) -> None:
    try:
        with conn.cursor() as cur:
            for stmt in stmts:
                cur.execute(stmt)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def fetch_one(conn, sql: str, params: tuple = ()):
    """Run a query, return one row (or None). Commits when done."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.commit()
        return row
    except Exception:
        conn.rollback()
        raise


def fetch_all(conn, sql: str, params: tuple = ()) -> list:
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise


def execute(conn, sql: str, params: tuple = ()) -> int:
    """Run INSERT/UPDATE/DELETE. Returns teh number of rows affected."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            count = cur.rowcount
        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise
