import httpx

from vechord.chunk import RegexChunker
from vechord.embedding import GeminiDenseEmbedding
from vechord.extract import SimpleExtractor
from vechord.registry import VechordRegistry
from vechord.rerank import CohereReranker
from vechord.spec import DefaultDocument, Keyword, create_chunk_with_dim

URL = "https://paulgraham.com/{}.html"
Chunk = create_chunk_with_dim(768)
emb = GeminiDenseEmbedding()
chunker = RegexChunker(size=1024, overlap=0)
reranker = CohereReranker()
extractor = SimpleExtractor()


vr = VechordRegistry("hybrid", "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([DefaultDocument, Chunk])


@vr.inject(output=DefaultDocument)
def load_document(title: str) -> DefaultDocument:
    with httpx.Client() as client:
        resp = client.get(URL.format(title))
        if resp.is_error:
            raise RuntimeError(f"Failed to fetch the document `{title}`")
        return DefaultDocument(title=title, text=extractor.extract_html(resp.text))


@vr.inject(input=DefaultDocument, output=Chunk)
def chunk_document(uid: int, text: str) -> list[Chunk]:
    chunks = chunker.segment(text)
    return [
        Chunk(
            doc_id=uid,
            text=chunk,
            vec=emb.vectorize_chunk(chunk),
            keyword=Keyword(chunk),
        )
        for chunk in chunks
    ]


def search_and_rerank(query: str, topk: int) -> list[Chunk]:
    text_retrieves = vr.search_by_keyword(Chunk, query, topk=topk)
    vec_retrievse = vr.search_by_vector(Chunk, emb.vectorize_query(query), topk=topk)
    chunks = list(
        {chunk.uid: chunk for chunk in text_retrieves + vec_retrievse}.values()
    )
    indices = reranker.rerank(query, [chunk.text for chunk in chunks])
    return [chunks[i] for i in indices[:topk]]


if __name__ == "__main__":
    load_document("smart")
    chunk_document()
    chunks = search_and_rerank("smart", 3)
    print(chunks)
