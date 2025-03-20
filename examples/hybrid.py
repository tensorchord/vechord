from html.parser import HTMLParser
from typing import Annotated

import httpx

from vechord.chunk import RegexChunker
from vechord.embedding import SpacyDenseEmbedding
from vechord.registry import VechordRegistry
from vechord.spec import ForeignKey, Keyword, PrimaryKeyAutoIncrease, Table, Vector

URL = "https://paulgraham.com/{}.html"
DenseVector = Vector[96]
emb = SpacyDenseEmbedding()
chunker = RegexChunker(size=1024, overlap=0)


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


class Document(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    title: str = ""
    text: str


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    doc_id: Annotated[int, ForeignKey[Document.uid]]
    text: str
    vector: DenseVector
    keyword: Keyword


vr = VechordRegistry("hybrid", "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Document, Chunk])


@vr.inject(output=Document)
def load_document(title: str) -> Document:
    with httpx.Client() as client:
        resp = client.get(URL.format(title))
        if resp.is_error:
            raise RuntimeError(f"Failed to fetch the document `{title}`")
    parser = EssayParser()
    parser.feed(resp.text)
    return Document(title=title, text="\n".join(t for t in parser.content if t))


@vr.inject(input=Document, output=Chunk)
def chunk_document(uid: int, text: str) -> list[Chunk]:
    chunks = chunker.segment(text)
    return [
        Chunk(
            doc_id=uid,
            text=chunk,
            vector=emb.vectorize_chunk(chunk),
            keyword=Keyword(chunk),
        )
        for chunk in chunks
    ]


if __name__ == "__main__":
    load_document("smart")
    chunk_document()

    vec = vr.search_by_vector(Chunk, emb.vectorize_query("smart"), topk=3)
    text = vr.search_by_keyword(Chunk, "smart", topk=3)

    from rich import print

    print(vec, text)
