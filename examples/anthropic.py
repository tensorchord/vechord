"""Anthropic Cookbook Contextual Embedding Example.

Data can be found from "https://github.com/anthropics/anthropic-cookbook".
"""

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import httpx

from vechord.augment import GeminiAugmenter
from vechord.embedding import GeminiDenseEmbedding
from vechord.registry import VechordRegistry
from vechord.rerank import CohereReranker, ReciprocalRankFusion
from vechord.spec import (
    ForeignKey,
    Keyword,
    PrimaryKeyAutoIncrease,
    Table,
    UniqueIndex,
    Vector,
)

DenseVector = Vector[3072]
emb = GeminiDenseEmbedding()


class Document(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    uuid: Annotated[str, UniqueIndex()]
    content: str


class Chunk(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    doc_uuid: Annotated[str, ForeignKey[Document.uuid]]
    index: int
    content: str
    vector: DenseVector
    keyword: Keyword


class ContextualChunk(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    doc_uuid: Annotated[str, ForeignKey[Document.uuid]]
    index: int
    content: str
    context: str
    vector: DenseVector
    keyword: Keyword


class Query(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    content: str
    answer: str
    doc_uuids: list[str]
    chunk_index: list[int]
    vector: DenseVector


vr = VechordRegistry(
    "anthropic",
    "postgresql://postgres:postgres@172.17.0.1:5432/",
    tables=[Document, Chunk, ContextualChunk, Query],
)


def download_data(url: str, save_path: str):
    if Path(save_path).is_file():
        print(f"{save_path} already exists, skip download.")
        return
    with httpx.stream("GET", url) as response, open(save_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)


async def load_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            await vr.insert(
                Document(
                    uuid=doc["original_uuid"],
                    content=doc["content"],
                )
            )
            for chunk in doc["chunks"]:
                await vr.insert(
                    Chunk(
                        doc_uuid=doc["original_uuid"],
                        index=chunk["original_index"],
                        content=chunk["content"],
                        vector=emb.vectorize_chunk(chunk["content"]),
                        keyword=Keyword(chunk["content"]),
                    )
                )


async def load_contextual_chunks(filepath: str):
    async with GeminiAugmenter() as augmenter:
        with open(filepath, "r", encoding="utf-8") as f:
            docs = json.load(f)
            for doc in docs:
                chunks = doc["chunks"]
                augments = await augmenter.augment_context(
                    doc["content"], [chunk["content"] for chunk in chunks]
                )
                if len(augments) != len(chunks):
                    print(
                        f"augments length not match for uuid: {doc['original_uuid']}, {len(augments)} != {len(chunks)}"
                    )
                for chunk, context in zip(chunks, augments, strict=False):
                    contextual_content = f"{chunk['content']}\n\n{context}"
                    await vr.insert(
                        ContextualChunk(
                            doc_uuid=doc["original_uuid"],
                            index=chunk["original_index"],
                            content=chunk["content"],
                            context=context,
                            vector=await emb.vectorize_chunk(contextual_content),
                            keyword=Keyword(contextual_content),
                        )
                    )


async def load_query(filepath: str):
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line)
            queries.append(
                Query(
                    content=query["query"],
                    answer=query["answer"],
                    doc_uuids=[x[0] for x in query["golden_chunk_uuids"]],
                    chunk_index=[x[1] for x in query["golden_chunk_uuids"]],
                    vector=await emb.vectorize_query(query["query"]),
                )
            )
    await vr.copy_bulk(queries)


async def vector_search(query: Query, topk: int) -> list[Chunk]:
    return await vr.search_by_vector(Chunk, query.vector, topk=topk)


async def vector_contextual_search(query: Query, topk: int) -> list[ContextualChunk]:
    return await vr.search_by_vector(ContextualChunk, query.vector, topk=topk)


async def keyword_search(query: Query, topk: int) -> list[Chunk]:
    return await vr.search_by_keyword(Chunk, query.content, topk=topk)


async def keyword_contextual_search(query: Query, topk: int) -> list[ContextualChunk]:
    return await vr.search_by_keyword(ContextualChunk, query.content, topk=topk)


async def hybrid_search_fuse(query: Query, topk: int) -> list[Chunk]:
    rrf = ReciprocalRankFusion()
    return rrf.fuse(
        [await vector_search(query, topk), await keyword_search(query, topk)]
    )[:topk]


async def hybrid_contextual_search_fuse(
    query: Query, topk: int
) -> list[ContextualChunk]:
    rrf = ReciprocalRankFusion()
    return rrf.fuse(
        [
            await vector_contextual_search(query, topk),
            await keyword_contextual_search(query, topk),
        ]
    )[:topk]


async def hybrid_search_rerank(query: Query, topk: int, boost=3) -> list[Chunk]:
    vecs = await vector_search(query, topk * boost)
    keys = await keyword_search(query, topk * boost)
    chunks = list({chunk.uid: chunk for chunk in vecs + keys}.values())
    async with CohereReranker() as ranker:
        indices = ranker.rerank(query.content, [chunk.content for chunk in chunks])
        return [chunks[i] for i in indices[:topk]]


async def hybrid_contextual_search_rerank(
    query: Query, topk: int, boost=3
) -> list[ContextualChunk]:
    vecs = await vector_contextual_search(query, topk * boost)
    keys = await keyword_contextual_search(query, topk * boost)
    chunks = list({chunk.uid: chunk for chunk in vecs + keys}.values())
    async with CohereReranker() as ranker:
        indices = ranker.rerank(
            query.content, [f"{chunk.content}\n{chunk.context}" for chunk in chunks]
        )
        return [chunks[i] for i in indices[:topk]]


async def evaluate(topk=5, search_func=vector_search):
    print(f"TopK={topk}, search by: {search_func.__name__}")
    queries: list[Query] = await vr.select_by(Query.partial_init())
    total_score = 0
    start = perf_counter()
    for query in queries:
        chunks: list[Chunk] = search_func(query, topk)
        count = 0
        for doc_uuid, chunk_index in zip(
            query.doc_uuids, query.chunk_index, strict=True
        ):
            for chunk in chunks:
                if chunk.doc_uuid == doc_uuid and chunk.index == chunk_index:
                    count += 1
                    break
        score = count / len(query.doc_uuids)
        total_score += score

    print(
        f"Pass@{topk}: {total_score / len(queries):.4f}, total queries: {len(queries)}, QPS: {len(queries) / (perf_counter() - start):.3f}"
    )


async def main(data_path: str):
    dir = Path(data_path)
    dir.mkdir(parents=True, exist_ok=True)
    download_data(
        "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/codebase_chunks.json",
        dir / "codebase_chunks.json",
    )
    download_data(
        "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl",
        dir / "evaluation_set.jsonl",
    )
    async with vr, emb:
        await load_data(dir / "codebase_chunks.json")
        await load_query(dir / "evaluation_set.jsonl")
        await load_contextual_chunks(dir / "codebase_chunks.json")

        for topk in [5, 10]:
            print("=" * 50)
            await evaluate(topk=topk, search_func=vector_search)
            await evaluate(topk=topk, search_func=keyword_search)
            await evaluate(topk=topk, search_func=hybrid_search_fuse)
            await evaluate(topk=topk, search_func=hybrid_search_rerank)
            await evaluate(topk=topk, search_func=vector_contextual_search)
            await evaluate(topk=topk, search_func=keyword_contextual_search)
            await evaluate(topk=topk, search_func=hybrid_contextual_search_fuse)
            await evaluate(topk=topk, search_func=hybrid_contextual_search_rerank)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main("datasets"))
