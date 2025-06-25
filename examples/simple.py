import asyncio
from typing import Optional

from vechord.embedding import GeminiDenseEmbedding
from vechord.registry import VechordRegistry
from vechord.spec import PrimaryKeyAutoIncrease, Table, Vector

DenseVector = Vector[3072]


class Document(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    title: str = ""
    text: str
    vec: DenseVector


async def main():
    async with (
        VechordRegistry(
            "simple",
            "postgresql://postgres:postgres@172.17.0.1:5432/",
            tables=[Document],
        ) as vr,
        GeminiDenseEmbedding() as emb,
    ):
        # add a document
        text = "my personal long note"
        doc = Document(
            title="note", text=text, vec=DenseVector(await emb.vectorize_chunk(text))
        )
        await vr.insert(doc)

        # load
        docs = await vr.select_by(Document.partial_init(), limit=1)
        print(docs)

        # query
        res = await vr.search_by_vector(
            Document, await emb.vectorize_query("note"), topk=1
        )
        print(res)

        # drop
        await vr.clear_storage(drop_table=True)


if __name__ == "__main__":
    asyncio.run(main())
