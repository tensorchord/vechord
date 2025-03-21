import contextlib
import contextvars
from typing import Any, Optional, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from vechord.spec import Keyword

active_cursor = contextvars.ContextVar("active_cursor", default=None)
select_transaction_buffer = contextvars.ContextVar(
    "select_transaction_buffer", default=False
)


@contextlib.contextmanager
def limit_to_transaction_buffer():
    """Only the rows inserted in the current transaction are returned."""
    token = select_transaction_buffer.set(True)
    try:
        yield
    finally:
        select_transaction_buffer.reset(token)


class VectorChordClient:
    def __init__(self, namespace: str, url: str):
        self.ns = namespace
        self.url = url
        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE")
        self.conn.execute('SET search_path TO "$user", public, bm25_catalog')
        register_vector(self.conn)

    @contextlib.contextmanager
    def transaction(self):
        with self.conn.transaction():
            cursor = self.conn.cursor()
            token = active_cursor.set(cursor)
            try:
                yield cursor
            finally:
                active_cursor.reset(token)

    def get_cursor(self):
        cursor = active_cursor.get()
        if cursor is not None:
            # in a transaction
            return cursor
        # single command auto-commit
        return self.conn.cursor()

    def create_table_if_not_exists(self, name: str, schema: Sequence[tuple[str, str]]):
        columns = sql.SQL(", ").join(
            sql.SQL("{col} {typ}").format(
                col=sql.Identifier(col),
                typ=sql.SQL(typ.format(namespace=self.ns)),
            )
            for col, typ in schema
        )
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {table} ({columns});").format(
                    table=sql.Identifier(f"{self.ns}_{name}"), columns=columns
                )
            )

    def create_vector_index(self, name: str, column: str):
        config = """
        residual_quantization = true
        [build.internal]
        lists = [1]
        spherical_centroids = false
        """
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON "
                    "{table} USING vchordrq ({column} vector_l2_ops) WITH "
                    "(options = $${config}$$);"
                ).format(
                    table=sql.Identifier(f"{self.ns}_{name}"),
                    index=sql.Identifier(f"{self.ns}_{name}_{column}_vec_idx"),
                    column=sql.Identifier(column),
                    config=sql.SQL(config),
                )
            )

    def create_multivec_index(self, name: str, column: str):
        config = "build.internal.lists = []"
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON "
                    "{table} USING vchordrq ({column} vector_maxsim_ip_ops) WITH "
                    "(options = $${config}$$);"
                ).format(
                    table=sql.Identifier(f"{self.ns}_{name}"),
                    index=sql.Identifier(f"{self.ns}_{name}_{column}_multivec_idx"),
                    column=sql.Identifier(column),
                    config=sql.SQL(config),
                )
            )

    def _keyword_index_name(self, name: str, column: str):
        return f"{self.ns}_{name}_{column}_bm25_idx"

    def create_keyword_index(self, name: str, column: str):
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON "
                    "{table} USING bm25 ({column} bm25_ops);"
                ).format(
                    table=sql.Identifier(f"{self.ns}_{name}"),
                    index=sql.Identifier(self._keyword_index_name(name, column)),
                    column=sql.Identifier(column),
                )
            )

    def select(
        self,
        name: str,
        raw_columns: Sequence[str],
        kvs: Optional[dict[str, Any]] = None,
        from_buffer: bool = False,
    ):
        """Select from db table with optional key-value condition or from un-committed
        transaction buffer.

        - `from_buffer`: this ensures the select query only returns the rows that are
            inserted in the current transaction.
        """
        columns = sql.SQL(", ").join(map(sql.Identifier, raw_columns))
        cursor = self.get_cursor()
        query = sql.SQL("SELECT {columns} FROM {table}").format(
            columns=columns,
            table=sql.Identifier(f"{self.ns}_{name}"),
        )
        if kvs:
            condition = sql.SQL(" AND ").join(
                sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder(col))
                for col in kvs
            )
            query += sql.SQL(" WHERE {condition}").format(condition=condition)
        elif from_buffer:
            query += sql.SQL(" WHERE xmin = pg_current_xact_id()::xid;")
        cursor.execute(query, kvs)
        return [row for row in cursor.fetchall()]

    @staticmethod
    def _to_placeholder(kv: tuple[str, Any]):
        key, value = kv
        if isinstance(value, Keyword):
            return sql.SQL("tokenize({}, {})").format(
                sql.Placeholder(key), sql.Literal(value._tokenizer)
            )
        return sql.Placeholder(key)

    def insert(self, name: str, values: dict):
        columns = sql.SQL(", ").join(map(sql.Identifier, values))
        placeholders = sql.SQL(", ").join(map(self._to_placeholder, values.items()))
        self.conn.execute(
            sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders});").format(
                table=sql.Identifier(f"{self.ns}_{name}"),
                columns=columns,
                placeholders=placeholders,
            ),
            values,
        )

    def delete(self, name: str, kvs: dict):
        if kvs:
            condition = sql.SQL(" AND ").join(
                sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder(col))
                for col in kvs
            )
            self.conn.execute(
                sql.SQL("DELETE FROM {table} WHERE {condition};").format(
                    table=sql.Identifier(f"{self.ns}_{name}"), condition=condition
                ),
                kvs,
            )
        else:
            self.conn.execute(
                sql.SQL("DELETE FROM {table};").format(
                    table=sql.Identifier(f"{self.ns}_{name}")
                )
            )

    def query_vec(
        self,
        name: str,
        vec_col: str,
        vec: np.ndarray,
        return_fields: list[str],
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        cursor = self.conn.execute(
            sql.SQL(
                "SELECT {columns} FROM {table} ORDER BY {vec_col} <-> %s LIMIT %s;"
            ).format(
                table=sql.Identifier(f"{self.ns}_{name}"),
                columns=columns,
                vec_col=sql.Identifier(vec_col),
            ),
            (vec, topk),
        )
        return [row for row in cursor.fetchall()]

    def query_multivec(  # noqa: PLR0913
        self,
        name: str,
        multivec_col: str,
        vec: np.ndarray,
        max_maxsim_tuples: int,
        return_fields: list[str],
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute("SET vchordrq.probes = '';")
            cursor.execute(
                sql.SQL("SET vchordrq.max_maxsim_tuples = {};").format(
                    sql.Literal(max_maxsim_tuples)
                )
            )
            cursor = cursor.execute(
                sql.SQL(
                    "SELECT {columns} FROM {table} ORDER BY {multivec_col} @# %s LIMIT %s;"
                ).format(
                    table=sql.Identifier(f"{self.ns}_{name}"),
                    columns=columns,
                    multivec_col=sql.Identifier(multivec_col),
                ),
                (vec, topk),
            )
            return [row for row in cursor.fetchall()]

    def query_keyword(  # noqa: PLR0913
        self,
        name: str,
        keyword_col: str,
        keyword: str,
        return_fields: Sequence[str],
        tokenizer: str,
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        cursor = self.conn.execute(
            sql.SQL(
                "SELECT {columns} FROM {table} ORDER BY {keyword_col} <&> "
                "to_bm25query({index}, %s, {tokenizer}) LIMIT %s;"
            ).format(
                table=sql.Identifier(f"{self.ns}_{name}"),
                columns=columns,
                index=sql.Literal(self._keyword_index_name(name, keyword_col)),
                tokenizer=sql.Literal(tokenizer),
                keyword_col=sql.Identifier(keyword_col),
            ),
            (keyword, topk),
        )
        return [row for row in cursor.fetchall()]

    def drop(self, name: str):
        self.conn.execute(
            sql.SQL("DROP TABLE IF EXISTS {table} CASCADE;").format(
                table=sql.Identifier(f"{self.ns}_{name}")
            )
        )
