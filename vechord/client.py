from time import perf_counter

import psycopg
from pgvector.psycopg import register_vector

from vechord.log import logger
from vechord.model import TextFile
from vechord.text import EN_TEXT_PROCESSOR


class VectorChordClient:
    def __init__(self, url: str, autocommit: bool = True):
        self.url = url
        self.conn = psycopg.connect(url, autocommit=autocommit)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        register_vector(self.conn)

    def create_namespace(self, namespace: str, dim: int = 96):
        config = """
        residual_quantization = true
        [build.internal]
        lists = [1]
        spherical_centroids = false
        """
        try:
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {namespace}_meta "
                "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "name TEXT, digest TEXT)"
            )
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {namespace} "
                "(id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                f"doc_id INT, content TEXT, embedding vector({dim}))"
            )
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS {namespace}_vector_idx ON {namespace} "
                "USING vchordrq (embedding vector_l2_ops) WITH "
                f"(options = $${config}$$)"
            )
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def insert_text(self, namespace: str, textfile: TextFile):
        try:
            cursor = self.conn.execute(
                f"INSERT INTO {namespace}_meta (name, digest) VALUES (%s, %s) RETURNING id",
                (textfile.filename, textfile.digest),
            )
            doc_id = cursor.fetchone()[0]
            for sentence in textfile.sentences:
                self.conn.execute(
                    f"INSERT INTO {namespace} (doc_id, content, embedding) VALUES (%s, %s, %s)",
                    (doc_id, sentence.text, sentence.vector),
                )
            logger.debug(
                "inserted %s sentences from file %s",
                len(textfile.sentences),
                textfile.filename,
            )
        except psycopg.errors.DatabaseError as err:
            logger.error(err)
            logger.info("rollback from the previous error")
            self.conn.rollback()
            raise err

    def query(self, namespace: str, query: str, topk: int = 10):
        start = perf_counter()
        query = EN_TEXT_PROCESSOR.process(query)
        try:
            cursor = self.conn.execute(
                f"SELECT content FROM {namespace} ORDER BY embedding <-> %s LIMIT %s",
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
