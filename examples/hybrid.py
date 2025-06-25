import httpx

from vechord.chunk import RegexChunker
from vechord.embedding import GeminiDenseEmbedding
from vechord.extract import SimpleExtractor
from vechord.registry import VechordRegistry
from vechord.rerank import CohereReranker
from vechord.spec import DefaultDocument, Keyword, create_chunk_with_dim

URL = "https://paulgraham.com/{}.html"
Chunk = create_chunk_with_dim(3072)
emb = GeminiDenseEmbedding()
chunker = RegexChunker(size=1024, overlap=0)
reranker = CohereReranker()
extractor = SimpleExtractor()


vr = VechordRegistry(
    "hybrid",
    "postgresql://postgres:postgres@172.17.0.1:5432/",
    tables=[DefaultDocument, Chunk],
)


@vr.inject(output=DefaultDocument)
async def load_document(title: str) -> DefaultDocument:
    async with httpx.AsyncClient() as client:
        resp = await client.get(URL.format(title))
        if resp.is_error:
            raise RuntimeError(f"Failed to fetch the document `{title}`")
        return DefaultDocument(title=title, text=extractor.extract_html(resp.text))


@vr.inject(input=DefaultDocument, output=Chunk)
async def chunk_document(uid: int, text: str) -> list[Chunk]:
    chunks = await chunker.segment(text)
    return [
        Chunk(
            doc_id=uid,
            text=chunk,
            vec=await emb.vectorize_chunk(chunk),
            keyword=Keyword(chunk),
        )
        for chunk in chunks
    ]


async def search_and_rerank(query: str, topk: int) -> list[Chunk]:
    text_retrieves = await vr.search_by_keyword(Chunk, query, topk=topk)
    vec_retrievse = await vr.search_by_vector(
        Chunk, await emb.vectorize_query(query), topk=topk
    )
    chunks = list(
        {chunk.uid: chunk for chunk in text_retrieves + vec_retrievse}.values()
    )
    indices = await reranker.rerank(query, [chunk.text for chunk in chunks])
    return [chunks[i] for i in indices[:topk]]


async def main():
    async with vr, emb, reranker:
        await load_document("smart")
        await chunk_document()
        chunks = await search_and_rerank("smart", 3)
        print(chunks)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
