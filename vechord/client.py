import contextlib
import math
from contextvars import ContextVar
from typing import Any, Optional, Sequence

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg.errors import DatabaseError
from psycopg_pool import AsyncConnectionPool

from vechord.spec import (
    AnyOf,
    IndexColumn,
    Keyword,
    KeywordIndex,
    MultiVectorIndex,
    UniqueIndex,
    VectorIndex,
)

DEFAULT_TOKENIZER = ("bert_base_uncased", "wiki_tocken")

select_transaction_buffer_conn = ContextVar[Optional[AsyncConnection]](
    "select_transaction_buffer_conn", default=None
)


@contextlib.asynccontextmanager
async def limit_to_transaction_buffer_conn(conn: AsyncConnection):
    """Only the rows inserted in the current transaction are returned."""
    token = select_transaction_buffer_conn.set(conn)
    try:
        yield
    finally:
        select_transaction_buffer_conn.reset(token)


current_namespace: ContextVar[Optional[str]] = ContextVar(
    "current_namespace", default=None
)


@contextlib.asynccontextmanager
async def set_namespace(ns: str):
    ns = current_namespace.set(ns)
    try:
        yield
    finally:
        current_namespace.reset(ns)


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
        self.pool = AsyncConnectionPool(
            self.url, open=False, configure=register_vector_async, max_size=16
        )

    @contextlib.asynccontextmanager
    async def get_connection(self):
        # reuse the connection in a custom transaction buffer
        if conn := select_transaction_buffer_conn.get():
            yield conn
            return

        async with self.pool.connection() as conn:
            yield conn

    def get_ns(self):
        """Get the current namespace.

        If it's not set by context manager, return the default namespace.
        """
        return current_namespace.get() or self.ns

    async def init_extension(self):
        """Initialize the required PostgreSQL extensions and set the search PATH.

        This should be called once before using the ConnectionPool since the pool
        requires the `vector` type to be created in the Database.
        """
        async with (
            await AsyncConnection.connect(self.url) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
            await cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25")
            await cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_tokenizer")
            await cursor.execute(
                'SET search_path TO "$user", public, bm25_catalog, tokenizer_catalog'
            )

    async def __aenter__(self):
        await self.init_extension()
        await self.pool.open()
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.pool.close()

    async def create_table_if_not_exists(
        self, name: str, schema: Sequence[tuple[str, str]]
    ):
        columns = sql.SQL(", ").join(
            sql.SQL("{col} {typ}").format(
                col=sql.Identifier(col),
                typ=sql.SQL(typ.format(namespace=self.get_ns())),
            )
            for col, typ in schema
        )
        async with self.get_connection() as conn:
            await conn.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {table} ({columns});").format(
                    table=sql.Identifier(f"{self.get_ns()}_{name}"), columns=columns
                )
            )

    async def create_tokenizer(self):
        async with self.get_connection() as conn:
            try:
                for name in DEFAULT_TOKENIZER:
                    await conn.execute(
                        sql.SQL(
                            "SELECT create_tokenizer({name}, $$model={model}$$)"
                        ).format(
                            name=sql.Literal(name),
                            model=sql.Identifier(name),
                        )
                    )
            except DatabaseError as err:
                if "already exists" not in str(err):
                    raise

    async def create_index_if_not_exists(self, name: str, column: IndexColumn):
        if isinstance(column.index, UniqueIndex):
            query = sql.SQL(
                "CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table} "
                "({column}) {config}"
            ).format(
                index_name=sql.Identifier(self._index_name(name, column)),
                table=sql.Identifier(f"{self.get_ns()}_{name}"),
                column=sql.Identifier(column.name),
                config=sql.SQL(column.index.config()),
            )
        else:
            query = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                "USING {index} ({column} {op_name})"
            ).format(
                index_name=sql.Identifier(self._index_name(name, column)),
                table=sql.Identifier(f"{self.get_ns()}_{name}"),
                index=sql.SQL(column.index.index),
                column=sql.Identifier(column.name),
                op_name=sql.SQL(column.index.op_name),
            )
            if config := column.index.config():
                query += sql.SQL(" WITH (options = $${config}$$)").format(
                    config=sql.SQL(config)
                )
        async with self.get_connection() as conn:
            await conn.execute(query)

    def _index_name(self, name: str, column: IndexColumn):
        return f"{self.get_ns()}_{name}_{column.name}_{column.index.name}"

    def _build_conditions(self, kvs: dict[str, Any]):
        if not kvs:
            return sql.SQL("TRUE")
        conditions = []
        for col, val in kvs.items():
            if val is None:
                conditions.append(sql.SQL("{} IS NULL").format(sql.Identifier(col)))
            elif isinstance(val, AnyOf):
                conditions.append(
                    sql.SQL("{} = ANY({})").format(
                        sql.Identifier(col), sql.Literal(val.values)
                    )
                )
            else:
                conditions.append(
                    sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder(col))
                )
        return sql.SQL(" AND ").join(conditions)

    async def select(
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
        query = sql.SQL("SELECT {columns} FROM {table}").format(
            columns=columns,
            table=sql.Identifier(f"{self.get_ns()}_{name}"),
        )
        if kvs:
            query += sql.SQL(" WHERE {condition}").format(
                condition=self._build_conditions(kvs)
            )
        elif from_buffer:
            query += sql.SQL(" WHERE xmin = pg_current_xact_id()::xid;")
        if limit:
            query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))

        async with self.get_connection() as conn:
            cursor = await conn.execute(query, kvs)
            return await cursor.fetchall()

    @staticmethod
    def _to_placeholder(kv: tuple[str, Any]):
        """Process the `Keyword` type"""
        key, value = kv
        if isinstance(value, Keyword):
            return sql.SQL("tokenize({}, {})").format(
                sql.Placeholder(key), sql.Literal(value._model)
            )
        return sql.Placeholder(key)

    async def insert(self, name: str, values: dict):
        columns = sql.SQL(", ").join(map(sql.Identifier, values))
        placeholders = sql.SQL(", ").join(map(self._to_placeholder, values.items()))
        query = sql.SQL(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders});"
        ).format(
            table=sql.Identifier(f"{self.get_ns()}_{name}"),
            columns=columns,
            placeholders=placeholders,
        )
        async with self.get_connection() as conn:
            await conn.execute(query, values)

    async def copy_bulk(self, name: str, values: Sequence[dict], types: Sequence[str]):
        columns = sql.SQL(", ").join(map(sql.Identifier, values[0]))
        query = sql.SQL(
            "COPY {table} ({columns}) FROM STDIN WITH (FORMAT BINARY)"
        ).format(
            table=sql.Identifier(f"{self.get_ns()}_{name}"),
            columns=columns,
        )
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            async with cursor.copy(query) as copy:
                copy.set_types(types=types)
                for value in values:
                    await copy.write_row(tuple(value.values()))

    async def delete(self, name: str, kvs: dict):
        async with self.get_connection() as conn:
            if kvs:
                await conn.execute(
                    sql.SQL("DELETE FROM {table} WHERE {condition};").format(
                        table=sql.Identifier(f"{self.get_ns()}_{name}"),
                        condition=self._build_conditions(kvs),
                    ),
                    kvs,
                )
            else:
                await conn.execute(
                    sql.SQL("DELETE FROM {table};").format(
                        table=sql.Identifier(f"{self.get_ns()}_{name}")
                    )
                )

    async def query_vec(  # noqa: PLR0913
        self,
        name: str,
        vec_col: IndexColumn[VectorIndex],
        vec: np.ndarray,
        return_fields: Sequence[str],
        topk: int = 10,
        probe: Optional[int] = None,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        if (
            probe is None
            and vec_col.index.lists is not None
            and vec_col.index.lists > 1
        ):
            probe = math.ceil(vec_col.index.lists / 16)
        set_probe = sql.SQL("SET LOCAL vchordrq.probes = {};").format(
            sql.Literal(probe or "")
        )
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            await cursor.execute(set_probe)
            await cursor.execute(
                sql.SQL(
                    "SELECT {columns} FROM {table} ORDER BY {vec_col} {op} %s LIMIT %s;"
                ).format(
                    table=sql.Identifier(f"{self.get_ns()}_{name}"),
                    columns=columns,
                    op=sql.SQL(vec_col.index.op_symbol),
                    vec_col=sql.Identifier(vec_col.name),
                ),
                (vec, topk),
            )
            return await cursor.fetchall()

    async def query_multivec(  # noqa: PLR0913
        self,
        name: str,
        multivec_col: IndexColumn[MultiVectorIndex],
        vec: np.ndarray,
        maxsim_refine: int,
        probe: Optional[int],
        return_fields: Sequence[str],
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        if (
            probe is None
            and multivec_col.index.lists is not None
            and multivec_col.index.lists > 1
        ):
            probe = math.ceil(multivec_col.index.lists / 16)
        set_probe = sql.SQL("SET LOCAL vchordrq.probes = {};").format(
            sql.Literal(probe or "")
        )
        set_refine = sql.SQL("SET LOCAL vchordrq.maxsim_refine = {};").format(
            sql.Literal(maxsim_refine)
        )
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            await cursor.execute(set_probe)
            await cursor.execute(set_refine)
            await cursor.execute(
                sql.SQL(
                    "SELECT {columns} FROM {table} ORDER BY {multivec_col} @# %s LIMIT %s;"
                ).format(
                    table=sql.Identifier(f"{self.get_ns()}_{name}"),
                    columns=columns,
                    multivec_col=sql.Identifier(multivec_col.name),
                ),
                (vec, topk),
            )
            return await cursor.fetchall()

    async def query_keyword(  # noqa: PLR0913
        self,
        name: str,
        keyword_col: IndexColumn[KeywordIndex],
        keyword: str,
        return_fields: Sequence[str],
        tokenizer: str,
        topk: int = 10,
    ):
        columns = sql.SQL(", ").join(map(sql.Identifier, return_fields))
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                sql.SQL(
                    "SELECT {columns} FROM {table} ORDER BY {keyword_col} <&> "
                    "to_bm25query({index}, tokenize(%s, {tokenizer})) LIMIT %s;"
                ).format(
                    table=sql.Identifier(f"{self.get_ns()}_{name}"),
                    columns=columns,
                    index=sql.Literal(self._index_name(name, keyword_col)),
                    tokenizer=sql.Literal(tokenizer),
                    keyword_col=sql.Identifier(keyword_col.name),
                ),
                (keyword, topk),
            )
            return await cursor.fetchall()

    async def drop(self, name: str):
        async with self.get_connection() as conn:
            await conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {table} CASCADE;").format(
                    table=sql.Identifier(f"{self.get_ns()}_{name}")
                )
            )
