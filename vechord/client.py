from time import perf_counter

import psycopg
from pgvector.psycopg import register_vector

from vechord.embedding import BaseEmbedding, VecType
from vechord.log import logger
from vechord.model import Chunk, Document

PSQL_NAMING_LIMIT = 64


class VectorChordClient:
    def __init__(self, namespace: str, url: str, autocommit: bool = True):
        self.ns = namespace
        self.url = url
        self.conn = psycopg.connect(url, autocommit=autocommit)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        register_vector(self.conn)

    def set_context(self, identifier: str, embs: list[BaseEmbedding]):
        self.identifier = identifier
        self.embs = embs
        self.chunk_table = f"{self.ns}_{identifier}"
        assert len(self.chunk_table) < PSQL_NAMING_LIMIT, (
            f"table name {self.chunk_table} too long"
        )

    def create(self):
        config = """
        residual_quantization = true
        [build.internal]
        lists = [1]
        spherical_centroids = false
        """
        try:
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.ns}_doc "
                "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "name TEXT, digest TEXT NOT NULL UNIQUE, updated_at TIMESTAMP"
                "identifier TEXT, data BYTEA)"
            )
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.chunk_table} "
                "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                f"doc_id INT, content TEXT))"
            )
            for emb in self.embs:
                if emb.vec_type != VecType.DENSE:
                    # only support dense embedding for now
                    continue
                table = f"{self.chunk_table}_{emb.name()}"
                assert len(table) < PSQL_NAMING_LIMIT, f"table name '{table}' too long"
                self.conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} "
                    f"(id INT, doc_id INT, embedding VECTOR({emb.get_dim()}))"
                )
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_vec_idx ON {table} "
                    "USING vchordrq (embedding vector_l2_ops) WITH "
                    f"(options = $${config}$$)"
                )
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def is_file_exists(self, doc: Document) -> bool:
        """
        This only checks if the file exists in the database and one of the chunks
        has all the configured embeddings.
        """
        cursor = self.conn.execute(
            f"SELECT id FROM {self.ns}_doc WHERE digest = %s", (doc.digest,)
        )
        if cursor.fetchone() is None:
            logger.debug("file %s not found in the doc table", doc.path)
            return False
        doc_id = cursor.fetchone()[0]
        cursor = self.conn.execute(
            f"SELECT id FROM {self.chunk_table} WHERE doc_id = %s limit 1", (doc_id,)
        )
        if cursor.fetchone() is None:
            logger.debug("file %s not found in the chunk table", doc.path)
            return False
        chunk_id = cursor.fetchone()[0]
        for emb in self.embs:
            table = f"{self.chunk_table}_{emb.name()}"
            cursor = self.conn.execute(
                f"SELECT id FROM {table} WHERE id = %s limit 1",
                (chunk_id,),
            )
            if cursor.fetchone() is None:
                logger.debug("file %s doesn't have emb %s", doc.path, emb.name())
                return False
        return True

    def insert_text(self, doc: Document, chunks: list[Chunk]):
        try:
            cursor = self.conn.execute(
                f"INSERT INTO {self.ns}_doc (name, digest, updated_at, identifier, data) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (doc.path, doc.digest, doc.updated_at, doc.identifier, doc.data),
            )
            doc_id = cursor.fetchone()[0]
            for chunk in chunks:
                cursor = self.conn.execute(
                    f"INSERT INTO {self.chunk_table} (doc_id, content) VALUES (%s, %s) RETURNING id",
                    (doc_id, chunk.text),
                )
                chunk_id = cursor.fetchone()[0]
                for emb in self.embs:
                    if emb.vec_type != VecType.DENSE:
                        continue
                    table = f"{self.chunk_table}_{emb.name()}"
                    cursor = self.conn.execute(
                        f"INSERT INTO {table} (id, doc_id, embedding) VALUES (%s, %s) RETURNING id",
                        (chunk_id, doc_id, emb.vectorize_chunk(chunk.text)),
                    )
            logger.debug("inserted %s sentences from file %s", len(chunks), doc.path)
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def delete(self, doc: Document):
        try:
            cursor = self.conn.execute(
                f"SELECT id FROM {self.ns}_doc WHERE digest = %s", (doc.digest,)
            )
            doc_id = cursor.fetchone()[0]
            cursor = self.conn.execute(
                f"DELETE FROM {self.chunk_table} WHERE doc_id = %s", (doc_id,)
            )
            cursor = self.conn.execute(
                f"DELETE FROM {self.ns}_doc WHERE id = %s", (doc_id,)
            )
            for emb in self.embs:
                table = f"{self.chunk_table}_{emb.name()}"
                cursor = self.conn.execute(
                    f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,)
                )
            logger.debug("deleted file %s", doc.path)
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def query(self, query: Chunk, topk: int = 10) -> list[str]:
        dense_emb = next(
            emb.name() for emb in self.embs if emb.vec_type == VecType.DENSE
        )
        assert dense_emb, "no dense embedding found"
        emb_table = f"{self.chunk_table}_{dense_emb}"
        start = perf_counter()
        try:
            cursor = self.conn.execute(
                "WITH ranked AS ("
                "SELECT c.id, c.content, e.embedding, "
                "ROW_NUMBER() OVER (ORDER BY e.embedding <-> %s) as ranking "
                f"FROM {self.chunk_table} c JOIN {emb_table} e ON c.id = e.id)"
                "SELECT id, content FROM ranked WHERE ranking <= %s",
                (query.vector, topk),
            )
            res = cursor.fetchall()
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err
        logger.debug("query time: %s", perf_counter() - start)
        return [row[0] for row in res]
