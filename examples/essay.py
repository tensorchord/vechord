from html.parser import HTMLParser
from typing import Annotated

import httpx

from vechord.chunk import RegexChunker
from vechord.embedding import GeminiDenseEmbedding
from vechord.evaluate import GeminiEvaluator
from vechord.registry import (
    ForeignKey,
    Memory,
    PrimaryKeyAutoIncrease,
    Table,
    VechordRegistry,
    Vector,
)

URL = "https://paulgraham.com/{}.html"
ARTICLE = "best"
TOP_K = 10


class EssayParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = ...) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        self.content = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.content.append(data.strip())


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    text: str
    vector: Vector[768]


class Query(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    cid: Annotated[int, ForeignKey[Chunk.uid]]
    text: str
    vector: Vector[768]


class Evaluation(Memory):
    map: float
    ndcg: float
    recall: float


vr = VechordRegistry(ARTICLE, "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Chunk, Query])
emb = GeminiDenseEmbedding()
evaluator = GeminiEvaluator()

with httpx.Client() as client:
    resp = client.get(URL.format(ARTICLE))
parser = EssayParser()
parser.feed(resp.text)
doc = "\n".join(t for t in parser.content if t)


@vr.inject(output=Chunk)
def segment_essay() -> list[Chunk]:
    chunker = RegexChunker()
    chunks = chunker.segment(doc)
    return [Chunk(text=chunk, vector=emb.vectorize_chunk(chunk)) for chunk in chunks]


@vr.inject(input=Chunk, output=Query)
def create_query(uid: int, text: str) -> Query:
    query = evaluator.produce_query(doc, text)
    return Query(cid=uid, text=query, vector=emb.vectorize_chunk(query))


@vr.inject(input=Query, output=Evaluation)
def evalute(cid: int, vector: Vector[768]) -> Evaluation:
    chunks: list[Chunk] = vr.search(Chunk, vector, topk=TOP_K)
    score = evaluator.evaluate_one(cid, [chunk.uid for chunk in chunks])
    return Evaluation(
        map=score["map"], ndcg=score["ndcg"], recall=score[f"recall_{TOP_K}"]
    )


if __name__ == "__main__":
    segment_essay()
    create_query()
    evalute()

    res: list[Evaluation] = vr.select_by(Evaluation, Evaluation.partial_init())
    print("ndcg", sum(r.ndcg for r in res) / len(res))
    print(f"recall@{TOP_K}", sum(r.recall for r in res) / len(res))
