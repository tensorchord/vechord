from datetime import datetime, timezone
from functools import partial
from typing import Annotated

import httpx
import msgspec

from vechord.chunk import RegexChunker
from vechord.embedding import GeminiDenseEmbedding
from vechord.extract import SimpleExtractor
from vechord.registry import VechordRegistry
from vechord.service import create_web_app
from vechord.spec import (
    ForeignKey,
    PrimaryKeyAutoIncrease,
    Table,
    Vector,
)

URL = "https://paulgraham.com/{}.html"
DenseVector = Vector[768]
emb = GeminiDenseEmbedding()
chunker = RegexChunker(size=1024, overlap=0)
extractor = SimpleExtractor()


class Document(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    title: str = ""
    text: str
    updated_at: datetime = msgspec.field(
        default_factory=partial(datetime.now, timezone.utc)
    )


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    doc_id: Annotated[int, ForeignKey[Document.uid]]
    text: str
    vector: DenseVector


vr = VechordRegistry("http", "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Document, Chunk])


@vr.inject(output=Document)
def load_document(title: str) -> Document:
    with httpx.Client() as client:
        resp = client.get(URL.format(title))
        if resp.is_error:
            raise RuntimeError(f"Failed to fetch the document `{title}`")
        return Document(title=title, text=extractor.extract_html(resp.text))


@vr.inject(input=Document, output=Chunk)
def chunk_document(uid: int, text: str) -> list[Chunk]:
    chunks = chunker.segment(text)
    return [
        Chunk(doc_id=uid, text=chunk, vector=DenseVector(emb.vectorize_chunk(chunk)))
        for chunk in chunks
    ]


if __name__ == "__main__":
    # this pipeline will be used in the web app, or you can run it with `vr.run()`
    pipeline = vr.create_pipeline([load_document, chunk_document])
    app = create_web_app(vr, pipeline)

    from wsgiref.simple_server import make_server

    with make_server("", 8000, app) as server:
        server.serve_forever()
