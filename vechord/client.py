from time import perf_counter

import psycopg
from pgvector.psycopg import register_vector

from vechord.log import logger
from vechord.model import Chunk, File


class VectorChordClient:
    def __init__(self, namespace: str, url: str, autocommit: bool = True):
        self.ns = namespace
        self.url = url
        self.conn = psycopg.connect(url, autocommit=autocommit)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        register_vector(self.conn)

    def create(self, dim):
        config = """
        residual_quantization = true
        [build.internal]
        lists = [1]
        spherical_centroids = false
        """
        try:
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.ns}_meta "
                "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "name TEXT, digest TEXT NOT NULL UNIQUE)"
            )
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.ns} "
                "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                f"doc_id INT, content TEXT, embedding vector({dim}))"
            )
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.ns}_vector_idx ON {self.ns} "
                "USING vchordrq (embedding vector_l2_ops) WITH "
                f"(options = $${config}$$)"
            )
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def is_file_exists(self, file: File) -> bool:
        cursor = self.conn.execute(
            f"SELECT id FROM {self.ns}_meta WHERE digest = %s", (file.digest,)
        )
        return cursor.fetchone() is not None

    def insert_text(self, file: File, chunks: list[Chunk]):
        try:
            cursor = self.conn.execute(
                f"INSERT INTO {self.ns}_meta (name, digest) VALUES (%s, %s) RETURNING id",
                (file.path, file.digest),
            )
            doc_id = cursor.fetchone()[0]
            for chunk in chunks:
                self.conn.execute(
                    f"INSERT INTO {self.ns} (doc_id, content, embedding) VALUES (%s, %s, %s)",
                    (doc_id, chunk.text, chunk.vector),
                )
            logger.debug("inserted %s sentences from file %s", len(chunks), file.path)
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def query(self, query: Chunk, topk: int = 10) -> list[str]:
        start = perf_counter()
        try:
            cursor = self.conn.execute(
                f"SELECT content FROM {self.ns} ORDER BY embedding <-> %s LIMIT %s",
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
