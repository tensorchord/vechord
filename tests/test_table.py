import asyncio
from datetime import datetime, timezone
from functools import partial
from typing import Annotated, Optional

import msgspec
import pytest
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from tests.conftest import DenseVector, gen_vector
from vechord.spec import (
    ForeignKey,
    Keyword,
    MultiVectorIndex,
    PrimaryKeyAutoIncrease,
    PrimaryKeyUUID,
    Table,
    UniqueIndex,
    VectorIndex,
)

pytestmark = pytest.mark.anyio


class Document(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    title: str = ""
    text: str
    updated_at: datetime = msgspec.field(
        default_factory=partial(datetime.now, timezone.utc)
    )


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    doc_id: Annotated[int, ForeignKey[Document.uid]]
    text: str
    vector: DenseVector
    keyword: Keyword


class AnnotatedChunk(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    text: str
    vector: Annotated[DenseVector, VectorIndex(distance="cos", lists=3)]


class Sentence(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    vector: list[DenseVector]


class UniqueTable(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    sid: Annotated[str, UniqueIndex()]


class SubTable(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    foreign_key: Annotated[str, ForeignKey[UniqueTable.sid]]


class JsonTable(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    data: Jsonb


class Image(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    vecs: Annotated[list[DenseVector], MultiVectorIndex(lists=2)]
    dataset: Optional[str] = None
    oid: Optional[int] = None


Tockenizer = Keyword.with_model("wiki_tocken")


class OtherTokenizer(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    keyword: Tockenizer


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Document,)], indirect=True)
async def test_insert_select_remove(registry):
    docs = [Document(text="hello world"), Document(text="hello there")]
    for doc in docs:
        await registry.insert(doc)

    # select all
    inserted = await registry.select_by(Document.partial_init(), fields=["text"])
    assert len(inserted) == len(docs)
    assert inserted[0].text == "hello world"
    assert inserted[1].text == "hello there"

    # select with limit
    one = await registry.select_by(Document.partial_init(), limit=1)
    assert len(one) == 1

    # select by id
    first = await registry.select_by(Document.partial_init(uid=1))
    assert len(first) == 1
    assert first[0].text == "hello world"

    # remove by id
    await registry.remove_by(Document.partial_init(uid=2))
    assert len(await registry.select_by(Document.partial_init())) == 1


@pytest.mark.db
@pytest.mark.parametrize("registry", [(AnnotatedChunk,)], indirect=True)
async def test_annotated_index(registry):
    num = 100
    topk = 5
    for text in (f"hello {i}" for i in range(num)):
        await registry.insert(AnnotatedChunk(text=text, vector=gen_vector()))

    inserted = await registry.select_by(AnnotatedChunk.partial_init(), fields=["text"])
    assert len(inserted) == num

    res = await registry.search_by_vector(AnnotatedChunk, gen_vector(), topk=topk)
    assert len(res) == topk


@pytest.mark.db
@pytest.mark.parametrize("registry", [(UniqueTable, SubTable)], indirect=True)
async def test_unique_index(registry):
    await registry.insert(UniqueTable(sid="id_0"))
    with pytest.raises(UniqueViolation):
        await registry.insert(UniqueTable(sid="id_0"))
    await registry.insert(SubTable(text="hello", foreign_key="id_0"))
    await registry.remove_by(UniqueTable.partial_init(sid="id_0"))
    assert len(await registry.select_by(UniqueTable.partial_init())) == 0
    assert len(await registry.select_by(SubTable.partial_init())) == 0


@pytest.mark.db
@pytest.mark.parametrize("registry", [(OtherTokenizer,)], indirect=True)
async def test_keyword_tokenizer(registry):
    num = 20
    topk = 5
    for text in (f"hello {i}" for i in range(num)):
        await registry.insert(OtherTokenizer(text=text, keyword=Tockenizer(text)))

    inserted = await registry.select_by(OtherTokenizer.partial_init(), fields=["text"])
    assert len(inserted) == num

    res = await registry.search_by_keyword(OtherTokenizer, "hello", topk=topk)
    assert len(res) == topk
    assert all("hello" in record.text for record in res)


@pytest.mark.db
@pytest.mark.parametrize("registry", [(JsonTable,)], indirect=True)
async def test_jsonb(registry):
    num = 10
    for i in range(num):
        await registry.insert(JsonTable(text=f"hello {i}", data=Jsonb({"key": i})))

    inserted = await registry.select_by(JsonTable.partial_init(), fields=["text"])
    assert len(inserted) == num


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Document, Chunk)], indirect=True)
async def test_foreign_key(registry):
    docs = [
        Document(text="hello world"),
        Document(text="hello there"),
    ]
    chunks = [
        Chunk(doc_id=1, text="hello", keyword=Keyword("hello"), vector=gen_vector()),
        Chunk(doc_id=1, text="world", keyword=Keyword("world"), vector=gen_vector()),
        Chunk(doc_id=1, text="no keyword field", keyword=None, vector=gen_vector()),
    ]
    for record in docs + chunks:
        await registry.insert(record)

    assert len(await registry.select_by(Document.partial_init())) == len(docs)
    assert len(await registry.select_by(Chunk.partial_init())) == len(chunks)
    # remove the doc should also remove the related chunks
    await registry.remove_by(Document.partial_init(uid=1))
    assert len(await registry.select_by(Document.partial_init())) == 1
    assert len(await registry.select_by(Chunk.partial_init())) == 0


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Document, Chunk)], indirect=True)
async def test_injection(registry):
    @registry.inject(output=Document)
    def create_doc(text: str) -> Document:
        return Document(text=text)

    @registry.inject(input=Document, output=Chunk)
    def create_chunk(uid: int, text: str) -> list[Chunk]:
        return [
            Chunk(doc_id=uid, text=t, keyword=Keyword(t), vector=gen_vector())
            for t in text.split()
        ]

    text = "hello world what happened to vector search"
    await create_doc(text)
    await create_chunk()

    docs = await registry.select_by(Document.partial_init())
    assert len(docs) == 1

    chunks = await registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(text.split())

    topk = 3
    # vector search
    vec_res = await registry.search_by_vector(Chunk, gen_vector(), topk=topk)
    assert len(vec_res) == topk
    assert all(chunk.text in text for chunk in vec_res)
    # keyword search
    text_res = await registry.search_by_keyword(Chunk, "vector", topk=topk)
    assert len(text_res) == 1


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Sentence,)], indirect=True)
async def test_multi_vec_maxsim(registry):
    @registry.inject(output=Sentence)
    def create_sentence(text: str) -> Sentence:
        return Sentence(
            text=text, vector=[gen_vector() for _ in range(len(text.split()))]
        )

    text = "the quick brown fox jumps over the lazy dog"
    num = 32
    for _ in range(num):
        await create_sentence(text)
    sentence = await registry.select_by(Sentence.partial_init())
    assert len(sentence) == num
    assert len(sentence[0].vector) == len(text.split())

    topk = 3
    for dim in range(1, 10):
        res = await registry.search_by_multivec(
            Sentence, [gen_vector() for _ in range(dim)], topk=topk
        )
        assert len(res) == topk


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Document, Chunk)], indirect=True)
async def test_pipeline(registry):
    @registry.inject(output=Document)
    def create_doc(text: str) -> Document:
        return Document(text=text)

    @registry.inject(input=Document, output=Chunk)
    def create_chunk(uid: int, text: str) -> list[Chunk]:
        nums = [int(x) for x in text.split()]
        return [
            Chunk(
                doc_id=uid,
                text=f"num[{num}]",
                keyword=Keyword(num),
                vector=gen_vector(),
            )
            for num in nums
        ]

    correct = "1 2 3 4 5"
    error = "100 0.1 no no"
    pipeline = registry.create_pipeline([create_doc, create_chunk])

    await pipeline.run(correct)
    docs = await registry.select_by(Document.partial_init())
    assert len(docs) == 1
    chunks = await registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(correct.split())

    # break the transaction won't add new records
    with pytest.raises(ValueError):
        await pipeline.run(error)

    docs = await registry.select_by(Document.partial_init())
    assert len(docs) == 1
    chunks = await registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(correct.split())


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Image,)], indirect=True)
async def test_multivec_copy(registry):
    contents = (
        "hello world",
        "hello there",
        "hi there, how are you",
        "the quick brown fox jumps over the lazy dog",
        "Across the Great Wall we can reach every corner in the world",
    )
    await registry.copy_bulk(
        [
            Image(text=text, vecs=[gen_vector() for _ in range(len(text.split()))])
            for text in contents
        ]
    )

    assert len(await registry.select_by(Image.partial_init())) == len(contents)

    await registry.copy_bulk(
        [
            Image(
                text="not bad as a test case",
                vecs=[gen_vector() for _ in range(len(contents))],
                dataset="test",
                oid=0,
            )
        ]
    )
    assert len(await registry.select_by(Image.partial_init())) == len(contents) + 1


@pytest.mark.db
@pytest.mark.parametrize("registry", [(Document, Chunk)], indirect=True)
async def test_search_return(registry):
    num = 100
    topk = 5
    await registry.insert(Document(text="hello world"))
    chunks = []
    for i in range(num):
        text = f"hello {i}"
        chunks.append(
            Chunk(doc_id=1, text=text, vector=gen_vector(), keyword=Keyword(text))
        )
    await asyncio.gather(*[registry.insert(chunk) for chunk in chunks])

    inserted: list[Chunk] = await registry.select_by(Chunk.partial_init(), fields=["text"])
    assert len(inserted) == num
    for record in inserted:
        assert record.text.startswith("hello")
        # vector field is not selected by default
        assert record.vector is msgspec.UNSET
        assert record.keyword is msgspec.UNSET

    res = await registry.search_by_vector(Chunk, gen_vector(), topk=topk)
    assert len(res) == topk
    assert all(record.vector is msgspec.UNSET for record in res)

    res = await registry.search_by_keyword(Chunk, "hello", topk=topk)
    assert len(res) == topk
    assert all(record.keyword is msgspec.UNSET for record in res)
