from datetime import datetime
from typing import Annotated, Optional

from rich import print

from vechord import (
    GeminiAugmenter,
    GeminiDenseEmbedding,
    GeminiEvaluator,
    LocalLoader,
    RegexChunker,
    SimpleExtractor,
)
from vechord.registry import VechordRegistry
from vechord.spec import (
    ForeignKey,
    PrimaryKeyAutoIncrease,
    Table,
    Vector,
)

emb = GeminiDenseEmbedding()
DenseVector = Vector[3072]
extractor = SimpleExtractor()


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
    vector: DenseVector


class ContextChunk(Table, kw_only=True):
    chunk_uid: Annotated[int, ForeignKey[Chunk.uid]]
    text: str
    vector: DenseVector


vr = VechordRegistry(
    "decorator",
    "postgresql://postgres:postgres@172.17.0.1:5432/",
    tables=[Document, Chunk, ContextChunk],
)


@vr.inject(output=Document)
def load_from_dir(dirpath: str) -> list[Document]:
    loader = LocalLoader(dirpath, include=[".pdf"])
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
async def split_document(uid: int, text: str) -> list[Chunk]:
    chunker = RegexChunker(overlap=0)
    chunks = await chunker.segment(text)
    return [
        Chunk(
            doc_uid=uid,
            seq_id=i,
            text=chunk,
            vector=DenseVector(await emb.vectorize_chunk(chunk)),
        )
        for i, chunk in enumerate(chunks)
    ]


@vr.inject(input=Document, output=ContextChunk)
async def context_embedding(uid: int, text: str) -> list[ContextChunk]:
    chunks: list[Chunk] = await vr.select_by(
        Chunk.partial_init(doc_uid=uid), fields=["uid", "text"]
    )
    async with GeminiAugmenter() as augmentor:
        context_chunks = [
            f"{context}\n{origin}"
            for (context, origin) in zip(
                augmentor.augment_context(text, [c.text for c in chunks]),
                [c.text for c in chunks],
                strict=False,
            )
        ]
    return [
        ContextChunk(
            chunk_uid=chunk_uid,
            text=augmented,
            vector=DenseVector(await emb.vectorize_chunk(augmented)),
        )
        for (chunk_uid, augmented) in zip(
            [c.uid for c in chunks], context_chunks, strict=False
        )
    ]


async def query_chunk(query: str) -> list[Chunk]:
    vector = await emb.vectorize_query(query)
    res: list[Chunk] = await vr.search_by_vector(Chunk, vector, topk=5)
    return res


async def query_context_chunk(query: str) -> list[ContextChunk]:
    vector = await emb.vectorize_query(query)
    res: list[ContextChunk] = await vr.search_by_vector(
        ContextChunk,
        vector,
        topk=5,
    )
    return res


@vr.inject(input=Chunk)
async def evaluate(uid: int, doc_uid: int, text: str):
    async with GeminiEvaluator() as evaluator:
        doc = (await vr.select_by(Document.partial_init(uid=doc_uid)))[0]
        query = await evaluator.produce_query(doc.text, text)
        retrieved = await query_chunk(query)
        score = evaluator.evaluate_one(str(uid), [str(r.uid) for r in retrieved])
    return score


async def main():
    async with vr, emb:
        await load_from_dir("./data")
        await split_document()
        await context_embedding()

        chunks = await query_chunk("vector search")
        print(chunks)

        scores = await evaluate()
        print(sum(scores) / len(scores))

        context_chunks = await query_context_chunk("vector search")
        print(context_chunks)

        await vr.clear_storage()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
