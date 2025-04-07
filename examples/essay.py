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

DenseVector = Vector[768]
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


vr = VechordRegistry(ARTICLE, "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Chunk, Query])

with httpx.Client() as client:
    resp = client.get(URL.format(ARTICLE))
doc = extractor.extract_html(resp.text)


@vr.inject(output=Chunk)
def segment_essay() -> list[Chunk]:
    chunker = RegexChunker()
    chunks = chunker.segment(doc)
    return [
        Chunk(text=chunk, vector=DenseVector(emb.vectorize_chunk(chunk)))
        for chunk in chunks
    ]


@vr.inject(input=Chunk, output=Query)
def create_query(uid: int, text: str) -> Query:
    query = evaluator.produce_query(doc, text)
    return Query(cid=uid, text=query, vector=DenseVector(emb.vectorize_chunk(query)))


@vr.inject(input=Query)
def evaluate(cid: int, vector: DenseVector) -> Evaluation:
    chunks: list[Chunk] = vr.search_by_vector(Chunk, vector, topk=TOP_K)
    score = evaluator.evaluate_one(str(cid), [str(chunk.uid) for chunk in chunks])
    return Evaluation(
        map=score["map"], ndcg=score["ndcg"], recall=score[f"recall_{TOP_K}"]
    )


if __name__ == "__main__":
    segment_essay()
    create_query()

    res: list[Evaluation] = evaluate()
    print("ndcg", sum(r.ndcg for r in res) / len(res))
    print(f"recall@{TOP_K}", sum(r.recall for r in res) / len(res))
