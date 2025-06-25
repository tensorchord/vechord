from dataclasses import dataclass
from typing import Annotated

import httpx

from vechord.chunk import RegexChunker
from vechord.embedding import GeminiDenseEmbedding
from vechord.evaluate import GeminiEvaluator
from vechord.extract import SimpleExtractor
from vechord.registry import VechordRegistry
from vechord.spec import (
    ForeignKey,
    PrimaryKeyAutoIncrease,
    Table,
    Vector,
)

URL = "https://paulgraham.com/{}.html"
ARTICLE = "best"
TOP_K = 10

DenseVector = Vector[3072]
emb = GeminiDenseEmbedding()
evaluator = GeminiEvaluator()
extractor = SimpleExtractor()


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    text: str
    vector: DenseVector


class Query(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    cid: Annotated[int, ForeignKey[Chunk.uid]]
    text: str
    vector: DenseVector


@dataclass(frozen=True)
class Evaluation:
    map: float
    ndcg: float
    recall: float


vr = VechordRegistry(
    ARTICLE, "postgresql://postgres:postgres@172.17.0.1:5432/", tables=[Chunk, Query]
)

with httpx.Client() as client:
    resp = client.get(URL.format(ARTICLE))
doc = extractor.extract_html(resp.text)


@vr.inject(output=Chunk)
async def segment_essay() -> list[Chunk]:
    chunker = RegexChunker()
    chunks = await chunker.segment(doc)
    return [
        Chunk(text=chunk, vector=DenseVector(await emb.vectorize_chunk(chunk)))
        for chunk in chunks
    ]


@vr.inject(input=Chunk, output=Query)
async def create_query(uid: int, text: str) -> Query:
    query = await evaluator.produce_query(doc, text)
    return Query(
        cid=uid, text=query, vector=DenseVector(await emb.vectorize_chunk(query))
    )


@vr.inject(input=Query)
async def evaluate(cid: int, vector: DenseVector) -> Evaluation:
    chunks: list[Chunk] = await vr.search_by_vector(Chunk, vector, topk=TOP_K)
    score = evaluator.evaluate_one(str(cid), [str(chunk.uid) for chunk in chunks])
    return Evaluation(
        map=score["map"], ndcg=score["ndcg"], recall=score[f"recall_{TOP_K}"]
    )


async def main():
    async with vr, emb, evaluator:
        await segment_essay()
        await create_query()

        res: list[Evaluation] = await evaluate()
        print("ndcg", sum(r.ndcg for r in res) / len(res))
        print(f"recall@{TOP_K}", sum(r.recall for r in res) / len(res))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
