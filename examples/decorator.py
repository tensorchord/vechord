from datetime import datetime
from typing import Annotated, Optional

from vechord import LocalLoader, RegexChunker, SimpleExtractor, SpacyDenseEmbedding
from vechord.registry import (
    ForeignKey,
    PrimaryKeyAutoIncrease,
    Table,
    VechordRegistry,
    Vector,
)


class Document(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    digest: str
    filename: str
    text: str
    updated_at: datetime


class Chunk(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    doc_uid: Annotated[int, ForeignKey[Document.uid]]
    seq_id: int
    text: str


class DenseEmbedding(Table, kw_only=True):
    chunk_uid: Annotated[int, ForeignKey[Chunk.uid]]
    vector: Vector[96]


vr = VechordRegistry("decorator", "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Document, Chunk, DenseEmbedding])


@vr.inject(output=Document)
def load_from_dir(dirpath: str) -> list[Document]:
    loader = LocalLoader(dirpath, include=[".pdf"])
    extractor = SimpleExtractor()
    return [
        Document(
            digest=doc.digest,
            filename=doc.path,
            text=extractor.extract(doc),
            updated_at=doc.updated_at,
        )
        for doc in loader.load()
    ]


@vr.inject(input=Document, output=Chunk)
def split_document(uid: int, text: str) -> list[Chunk]:
    chunker = RegexChunker(overlap=0)
    chunks = chunker.segment(text)
    return [Chunk(doc_uid=uid, seq_id=i, text=chunk) for i, chunk in enumerate(chunks)]


dense = SpacyDenseEmbedding()


@vr.inject(input=Chunk, output=DenseEmbedding)
def embed_chunk(uid: int, text: str) -> DenseEmbedding:
    vector = dense.vectorize_chunk(text)
    return DenseEmbedding(chunk_uid=uid, vector=vector)


load_from_dir("./data")
split_document()
embed_chunk()
vr.clear_storage()
