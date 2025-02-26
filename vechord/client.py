import hashlib
from typing import Any, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from vechord.augment import BaseAugmenter
from vechord.embedding import BaseEmbedding, VecType
from vechord.log import logger
from vechord.model import Chunk, Document, RetrievedChunk

PSQL_NAMING_LIMIT = 64


def hash_table_suffix(name: str) -> str:
    return hashlib.shake_256(name.encode()).hexdigest(4)


class VectorChordClient:
    def __init__(self, namespace: str, url: str, autocommit: bool = True):
        self.ns = namespace
        self.url = url
        self.conn = psycopg.connect(url, autocommit=autocommit)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        register_vector(self.conn)

    def create_table_if_not_exists(self, name: str, schema: list[tuple[str, str]]):
        cursor = self.conn.cursor()
        columns = ", ".join(
            f"{col} {typ.format(namespace=self.ns)}" for col, typ in schema
        )
        with self.conn.transaction():
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.ns}_{name} ({columns});")

    def create_vector_index(self, name: str, column: str):
        config = """
        residual_quantization = true
        [build.internal]
        lists = [1]
        spherical_centroids = false
        """
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.ns}_{name}_{column}_vec_idx ON "
            f"{self.ns}_{name} USING vchordrq ({column} vector_l2_ops) WITH "
            f"(options = $${config}$$);"
        )

    def select(
        self,
        name: str,
        columns: list[str],
        kvs: Optional[dict[str, Any]] = None,
    ):
        columns = ", ".join(columns)
        if kvs:
            condition = " AND ".join(f"{col} = %({col})s" for col in kvs)
            cursor = self.conn.execute(
                f"SELECT {columns} FROM {self.ns}_{name} WHERE {condition};", kvs
            )
        else:
            cursor = self.conn.execute(f"SELECT {columns} FROM {self.ns}_{name};")
        return [row for row in cursor.fetchall()]

    def insert(self, name: str, values: dict):
        columns = ", ".join(values.keys())
        placeholders = ", ".join(f"%({col})s" for col in values)
        self.conn.execute(
            f"INSERT INTO {self.ns}_{name} ({columns}) VALUES ({placeholders});", values
        )

    def delete(self, name: str, kvs: dict):
        if kvs:
            condition = " AND ".join(f"{col} = %({col})s" for col in kvs)
            self.conn.execute(f"DELETE FROM {self.ns}_{name} WHERE {condition};", kvs)
        else:
            self.conn.execute(f"DELETE FROM {self.ns}_{name};")

    def query_vec(
        self,
        name: str,
        vec_col: str,
        vec: np.ndarray,
        return_fields: list[str],
        topk: int = 10,
    ):
        columns = ", ".join(return_fields)
        cursor = self.conn.execute(
            f"SELECT {columns} FROM {self.ns}_{name} ORDER BY {vec_col} <-> %s LIMIT %s;",
            (vec, topk),
        )
        return [row for row in cursor.fetchall()]

    def drop(self, name: str):
        self.conn.execute(f"DROP TABLE IF EXISTS {self.ns}_{name} CASCADE;")

    def _get_emb_table_name(self, emb: BaseEmbedding) -> str:
        return f"{self.ns}_emb_{hash_table_suffix(self.chunk_table + emb.name())}"

    def set_context(
        self,
        identifier: str,
        embs: list[BaseEmbedding],
        augmenter: Optional[BaseAugmenter],
    ):
        self.identifier = identifier
        self.embs = embs
        self.augmenter = augmenter
        self.chunk_table = f"{self.ns}_chunk_{hash_table_suffix(identifier)}"
        if self.augmenter:
            self.chunk_table = f"{self.ns}_aug_chunk_{hash_table_suffix(identifier + self.augmenter.name())}"
        assert len(self.chunk_table) < PSQL_NAMING_LIMIT, (
            f"table name {self.chunk_table} too long"
        )
        self._context_table_exists: Optional[bool] = None

    def create(self):
        config = """
        residual_quantization = true
        [build.internal]
        lists = [1]
        spherical_centroids = false
        """
        try:
            cursor = self.conn.cursor()
            with self.conn.transaction():
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.ns}_doc "
                    "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                    "name TEXT, digest TEXT NOT NULL UNIQUE, updated_at TIMESTAMP, "
                    "source TEXT, data BYTEA);"
                )
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.chunk_table} "
                    "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                    f"doc_id INT, seq_id INT, content TEXT);"
                    f"COMMENT ON TABLE {self.chunk_table} IS 'from {self.identifier}';"
                )
                for emb in self.embs:
                    if emb.vec_type() != VecType.DENSE:
                        # only support dense embedding for now
                        continue
                    table = self._get_emb_table_name(emb)
                    assert len(table) < PSQL_NAMING_LIMIT, (
                        f"table name '{table}' too long"
                    )
                    cursor.execute(
                        f"CREATE TABLE IF NOT EXISTS {table} "
                        f"(id INT NOT NULL UNIQUE, doc_id INT, embedding VECTOR({emb.get_dim()}));"
                        f"COMMENT ON TABLE {table} IS 'from {self.identifier}-{emb.name()}';"
                    )
                    cursor.execute(
                        f"CREATE INDEX IF NOT EXISTS {table}_vec_idx ON {table} "
                        "USING vchordrq (embedding vector_l2_ops) WITH "
                        f"(options = $${config}$$);"
                    )
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def is_table_exists(self) -> bool:
        if self._context_table_exists is not None:
            return self._context_table_exists

        cursor = self.conn.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            f"WHERE table_name = '{self.ns}_doc')"
        )
        doc_exists = cursor.fetchone()
        if doc_exists is None or doc_exists[0] is False:
            self._context_table_exists = False
            return False

        cursor = self.conn.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            f"WHERE table_name = '{self.chunk_table}')"
        )
        chunk_exists = cursor.fetchone()
        if chunk_exists is None or chunk_exists[0] is False:
            self._context_table_exists = False
            return False

        for emb in self.embs:
            table = self._get_emb_table_name(emb)
            cursor = self.conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                f"WHERE table_name = '{table}')"
            )
            emb_exists = cursor.fetchone()
            if emb_exists is None or emb_exists[0] is False:
                self._context_table_exists = False
                return False
        return True

    def is_file_exists(self, doc: Document) -> bool:
        """
        This only checks if the file exists in the database and one of the chunks
        has all the configured embeddings.
        """
        if not self.is_table_exists():
            return

        cursor = self.conn.execute(
            f"SELECT id FROM {self.ns}_doc WHERE digest = %s", (doc.digest,)
        )
        doc_id = cursor.fetchone()
        if doc_id is None:
            logger.debug("file `%s` not found in the doc table", doc.path)
            return False
        doc_id = doc_id[0]
        cursor = self.conn.execute(
            f"SELECT id FROM {self.chunk_table} WHERE doc_id = %s limit 1", (doc_id,)
        )
        chunk_id = cursor.fetchone()
        if chunk_id is None:
            logger.debug("file `%s` not found in the chunk table", doc.path)
            return False
        chunk_id = chunk_id[0]
        for emb in self.embs:
            table = self._get_emb_table_name(emb)
            cursor = self.conn.execute(
                f"SELECT id FROM {table} WHERE id = %s limit 1",
                (chunk_id,),
            )
            if cursor.fetchone() is None:
                logger.debug("file `%s` doesn't have emb %s", doc.path, emb.name())
                return False
        return True

    def insert_doc(self, doc: Document, chunks: list[Chunk]):
        try:
            cursor = self.conn.cursor()
            with self.conn.transaction():
                cursor.execute(
                    f"INSERT INTO {self.ns}_doc (name, digest, updated_at, source, data) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (digest) DO UPDATE SET name = {self.ns}_doc.name RETURNING id",
                    (doc.path, doc.digest, doc.updated_at, doc.source, doc.data),
                )
                doc_id = cursor.fetchone()[0]

                for chunk in chunks:
                    cursor.execute(
                        f"INSERT INTO {self.chunk_table} (doc_id, seq_id, content) VALUES (%s, %s, %s) RETURNING id",
                        (doc_id, chunk.seq_id, chunk.text),
                    )
                    chunk_id = cursor.fetchone()[0]
                    for emb in self.embs:
                        if emb.vec_type() != VecType.DENSE:
                            continue
                        table = self._get_emb_table_name(emb)
                        cursor.execute(
                            f"INSERT INTO {table} (id, doc_id, embedding) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                            (chunk_id, doc_id, emb.vectorize_chunk(chunk.text)),
                        )
            logger.debug("inserted %s chunks from file `%s`", len(chunks), doc.path)
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def delete_doc(self, doc: Document):
        try:
            cursor = self.conn.cursor()
            with self.conn.transaction():
                cursor.execute(
                    f"SELECT id FROM {self.ns}_doc WHERE digest = %s", (doc.digest,)
                )
                doc_id = cursor.fetchone()[0]
                cursor.execute(
                    f"DELETE FROM {self.chunk_table} WHERE doc_id = %s", (doc_id,)
                )
                cursor.execute(f"DELETE FROM {self.ns}_doc WHERE id = %s", (doc_id,))
                for emb in self.embs:
                    table = self._get_emb_table_name(emb)
                    cursor.execute(f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,))
            logger.debug("deleted file %s", doc.path)
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def query_chunk(self, query: Chunk, topk: int = 10) -> list[RetrievedChunk]:
        dense_emb = next(emb for emb in self.embs if emb.vec_type() == VecType.DENSE)
        assert dense_emb, "no dense embedding found"
        emb_table = self._get_emb_table_name(dense_emb)
        try:
            cursor = self.conn.execute(
                "SELECT c.id, c.content, e.embedding <-> %s as score "
                f"FROM {self.chunk_table} c JOIN {emb_table} e ON c.id = e.id "
                "ORDER BY score LIMIT %s",
                (dense_emb.vectorize_query(query.text), topk),
            )
            res = cursor.fetchall()
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err
        return [RetrievedChunk(uid=row[0], text=row[1], score=row[2]) for row in res]
