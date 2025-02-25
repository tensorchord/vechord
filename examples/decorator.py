from datetime import datetime
from typing import Annotated, Optional

from vechord import (
    GeminiAugmenter,
    GeminiEvaluator,
    LocalLoader,
    RegexChunker,
    SimpleExtractor,
    SpacyDenseEmbedding,
)
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
    vector: Vector[96]


class ContextChunk(Table, kw_only=True):
    chunk_uid: Annotated[int, ForeignKey[Chunk.uid]]
    text: str
    vector: Vector[96]


vr = VechordRegistry("decorator", "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Document, Chunk, ContextChunk])


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


dense = SpacyDenseEmbedding()


@vr.inject(input=Document, output=Chunk)
def split_document(uid: int, text: str) -> list[Chunk]:
    chunker = RegexChunker(overlap=0)
    chunks = chunker.segment(text)
    return [
        Chunk(doc_uid=uid, seq_id=i, text=chunk, vector=dense.vectorize_chunk(chunk))
        for i, chunk in enumerate(chunks)
    ]


@vr.inject(input=Chunk, output=ContextChunk)
def context_embedding(uid: int, doc_uid: int, text: str) -> ContextChunk:
    doc: Document = vr.select_from_storage(Document, uid, doc_uid)[0]
    augmentor = GeminiAugmenter()
    augmentor.reset(doc.text)
    context = augmentor.augment_context([text])[0]
    context_chunk = "\n".join([context, text])
    return ContextChunk(
        chunk_uid=uid, text=context_chunk, vector=dense.vectorize_chunk(context_chunk)
    )


def query_chunk(query: str) -> list[Chunk]:
    vector = dense.vectorize_query(query)
    res: list[Chunk] = vr.search(
        Chunk,
        vector,
        topk=5,
        return_vector=False,
    )
    return res


def query_context_chunk(query: str) -> list[ContextChunk]:
    vector = dense.vectorize_query(query)
    res: list[ContextChunk] = vr.search(
        ContextChunk,
        vector,
        topk=5,
        return_vector=False,
    )
    return res


@vr.inject(input=Chunk)
def evaluate(uid: int, doc_uid: int, text: str):
    evaluator = GeminiEvaluator()
    doc: Document = vr.select_from_storage(Document, doc_uid)[0]
    query = evaluator.produce_query(doc.text, text)
    retrieved = query_chunk(query)
    score = evaluator.evaluate_one(uid, [r.uid for r in retrieved])
    return score


if __name__ == "__main__":
    from rich import print

    load_from_dir("./data")
    split_document()
    # context_embedding()

    chunks = query_chunk("vector search")
    print(chunks)

    vr.clear_storage()
