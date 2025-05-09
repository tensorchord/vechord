"""Anthropic Cookbook Contextual Embedding Example.

Data can be found from "https://github.com/anthropics/anthropic-cookbook".
"""

import json
from typing import Annotated, Optional

import httpx

from vechord.augment import GeminiAugmenter
from vechord.embedding import SpacyDenseEmbedding
from vechord.registry import VechordRegistry
from vechord.spec import (
    ForeignKey,
    Keyword,
    PrimaryKeyAutoIncrease,
    Table,
    UniqueIndex,
    Vector,
)

DenseVector = Vector[96]
emb = SpacyDenseEmbedding()
vr = VechordRegistry("anthropic", "postgresql://postgres:postgres@172.17.0.1:5432/")


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


vr.register([Document, Chunk, ContextualChunk, Query])


def download_data(url: str, save_path: str):
    with httpx.stream("GET", url) as response, open(save_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)


def load_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            vr.insert(
                Document(
                    uuid=doc["original_uuid"],
                    content=doc["content"],
                )
            )
            for chunk in doc["chunks"]:
                vr.insert(
                    Chunk(
                        doc_uuid=doc["original_uuid"],
                        index=chunk["original_index"],
                        content=chunk["content"],
                        vector=emb.vectorize_chunk(chunk["content"]),
                        keyword=Keyword(chunk["content"]),
                    )
                )


def load_contextual_chunks(filepath: str):
    augmenter = GeminiAugmenter()

    with open(filepath, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            vr.insert(
                Document(
                    uuid=doc["original_uuid"],
                    content=doc["content"],
                )
            )
            augmenter.reset(doc["content"])
            context = augmenter.augment_context(
                [chunk["content"] for chunk in doc["chunks"]]
            )
            for i, chunk in enumerate(doc["chunks"]):
                contextual_content = f"{chunk['content']}\n\n{context[i]}"
                vr.insert(
                    ContextualChunk(
                        doc_uuid=doc["original_uuid"],
                        index=chunk["original_index"],
                        content=chunk["content"],
                        context=context[i],
                        vector=emb.vectorize_chunk(contextual_content),
                        keyword=Keyword(contextual_content),
                    )
                )


def load_query(filepath: str):
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
                    vector=emb.vectorize_query(query["query"]),
                )
            )
    vr.copy_bulk(queries)


def vector_search(query: Query, topk: int) -> list[Chunk]:
    return vr.search_by_vector(Chunk, query.vector, topk=topk)


def hybrid_search(query: Query, topk: int) -> list[Chunk]:
    kws = vr.search_by_keyword(Chunk, query.content, topk=topk)
    return kws
    # vecs = vr.search_by_vector(Chunk, query.vector, topk=topk)
    # rrf = ReciprocalRankFusion()
    # return rrf.fuse([vecs, kws])[:topk]


def evaluate(topk=5, search_func=vector_search):
    queries: list[Query] = vr.select_by(Query.partial_init())
    total_score = 0
    for query in queries:
        chunks: list[Chunk] = search_func(query, topk)
        count = 0
        # print("expect:", query.doc_uuids, query.chunk_index)
        # print("retrieved:", [(chunk.doc_uuid, chunk.index) for chunk in chunks])
        for doc_uuid, chunk_index in zip(
            query.doc_uuids, query.chunk_index, strict=True
        ):
            for chunk in chunks:
                if chunk.doc_uuid == doc_uuid and chunk.index == chunk_index:
                    count += 1
                    break
        score = count / len(query.doc_uuids)
        total_score += score

    print(f"Pass@{topk}: {total_score / len(queries)}, total queries: {len(queries)}")


if __name__ == "__main__":
    # download_data(
    #     "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/codebase_chunks.json",
    #     "datasets/codebase_chunks.json",
    # )
    # download_data(
    #     "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl",
    #     "datasets/evaluation_set.jsonl",
    # )
    # load_data("datasets/codebase_chunks.json")
    # load_contextual_chunks("datasets/codebase_chunks.json")
    # load_query("datasets/evaluation_set.jsonl")
    evaluate(search_func=hybrid_search)
