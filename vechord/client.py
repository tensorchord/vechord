import contextlib
import contextvars
from typing import Any, Optional, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.pq import TransactionStatus

from vechord.spec import IndexColumn, Keyword

DEFAULT_TOKENIZER = ("bert_base_uncased", "wiki_tocken")

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


class VechordClient:
    """A PostgreSQL client to access the database.

    Args:
        namespace: used as a prefix for the table name.
        url: the database connection URL.
            e.g. "postgresql://user:password@localhost:5432/dbname"
    """

    def __init__(self, namespace: str, url: str):
        self.ns = namespace
        self.url = url
        self.conn = psycopg.connect(url, autocommit=True)
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_tokenizer")
            cursor.execute(
                'SET search_path TO "$user", public, bm25_catalog, tokenizer_catalog'
            )
        register_vector(self.conn)

    @contextlib.contextmanager
    def transaction(self):
        """Create a transaction context manager (when there is no transaction)."""
        if self.conn.info.transaction_status != TransactionStatus.IDLE:
            yield None
            return

        with self.conn.transaction():
            cursor = self.conn.cursor()
            token = active_cursor.set(cursor)
            try:
                yield cursor
            finally:
                active_cursor.reset(token)

    def get_cursor(self):
        """Get the current cursor or create a new one."""
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

    def create_tokenizer(self):
        with self.transaction():
            cursor = self.get_cursor()
            try:
                for name in DEFAULT_TOKENIZER:
                    cursor.execute(
                        sql.SQL(
                            "SELECT create_tokenizer({name}, $$model={model}$$)"
                        ).format(
                            name=sql.Literal(name),
                            model=sql.Identifier(name),
                        )
                    )
            except psycopg.errors.DatabaseError as err:
                if "already exists" not in str(err):
                    raise

    def create_index_if_not_exists(self, name: str, column: IndexColumn):
        with self.transaction():
            cursor = self.get_cursor()
            query = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                "USING {index} ({column} {op_name})"
            ).format(
                index_name=sql.Identifier(self._index_name(name, column)),
                table=sql.Identifier(f"{self.ns}_{name}"),
                index=sql.SQL(column.index.index),
                column=sql.Identifier(column.name),
                op_name=sql.SQL(column.index.op_name),
            )
            if config := column.index.config():
                query += sql.SQL(" WITH (options = $${config}$$)").format(
                    config=sql.SQL(config)
                )
            cursor.execute(query)

    def _index_name(self, name: str, column: IndexColumn):
        return f"{self.ns}_{name}_{column.name}_{column.index.name}"

    def select(
        self,
        name: str,
        raw_columns: Sequence[str],
        kvs: Optional[dict[str, Any]] = None,
        from_buffer: bool = False,
        limit: Optional[int] = None,
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
                sql.SQL("{} IS NULL").format(sql.Identifier(col))
                if val is None
                else sql.SQL("{} = {}").format(
                    sql.Identifier(col), sql.Placeholder(col)
                )
                for col, val in kvs.items()
            )
            query += sql.SQL(" WHERE {condition}").format(condition=condition)
        elif from_buffer:
            query += sql.SQL(" WHERE xmin = pg_current_xact_id()::xid;")
        if limit:
            query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
        cursor.execute(query, kvs)
        return [row for row in cursor.fetchall()]

    @staticmethod
    def _to_placeholder(kv: tuple[str, Any]):
        """Process the `Keyword` type"""
        key, value = kv
        if isinstance(value, Keyword):
            return sql.SQL("tokenize({}, {})").format(
                sql.Placeholder(key), sql.Literal(value._model)
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

    def copy_bulk(self, name: str, values: Sequence[dict]):
        columns = sql.SQL(", ").join(map(sql.Identifier, values[0]))
        with self.transaction():
            cursor = self.get_cursor()
            with cursor.copy(
                sql.SQL(
                    "COPY {table} ({columns}) FROM STDIN WITH (FORMAT BINARY)"
                ).format(
                    table=sql.Identifier(f"{self.ns}_{name}"),
                    columns=columns,
                )
            ) as copy:
                for value in values:
                    copy.write_row(tuple(value.values()))

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
        vec_col: IndexColumn,
        vec: np.ndarray,
        return_fields: Sequence[str],
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        cursor = self.conn.execute(
            sql.SQL(
                "SELECT {columns} FROM {table} ORDER BY {vec_col} {op} %s LIMIT %s;"
            ).format(
                table=sql.Identifier(f"{self.ns}_{name}"),
                columns=columns,
                op=sql.SQL(vec_col.index.op_symbol),
                vec_col=sql.Identifier(vec_col.name),
            ),
            (vec, topk),
        )
        return [row for row in cursor.fetchall()]

    def query_multivec(  # noqa: PLR0913
        self,
        name: str,
        multivec_col: IndexColumn,
        vec: np.ndarray,
        max_maxsim_tuples: int,
        probe: Optional[int],
        return_fields: Sequence[str],
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        with self.transaction():
            cursor = self.get_cursor()
            cursor.execute(
                sql.SQL("SET LOCAL vchordrq.probes = {};").format(
                    sql.Literal(probe or "")
                )
            )
            cursor.execute(
                sql.SQL("SET LOCAL vchordrq.max_maxsim_tuples = {};").format(
                    sql.Literal(max_maxsim_tuples)
                )
            )
            cursor.execute(
                sql.SQL(
                    "SELECT {columns} FROM {table} ORDER BY {multivec_col} @# %s LIMIT %s;"
                ).format(
                    table=sql.Identifier(f"{self.ns}_{name}"),
                    columns=columns,
                    multivec_col=sql.Identifier(multivec_col.name),
                ),
                (vec, topk),
            )
            return [row for row in cursor.fetchall()]

    def query_keyword(  # noqa: PLR0913
        self,
        name: str,
        keyword_col: IndexColumn,
        keyword: str,
        return_fields: Sequence[str],
        tokenizer: str,
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        cursor = self.conn.execute(
            sql.SQL(
                "SELECT {columns} FROM {table} ORDER BY {keyword_col} <&> "
                "to_bm25query({index}, tokenize(%s, {tokenizer})) LIMIT %s;"
            ).format(
                table=sql.Identifier(f"{self.ns}_{name}"),
                columns=columns,
                index=sql.Literal(self._index_name(name, keyword_col)),
                tokenizer=sql.Literal(tokenizer),
                keyword_col=sql.Identifier(keyword_col.name),
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
