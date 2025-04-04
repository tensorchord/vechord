from datetime import datetime
from os import environ
from typing import Annotated, Optional

import msgspec
import numpy as np
import pytest

from vechord.log import logger
from vechord.registry import VechordRegistry
from vechord.spec import (
    ForeignKey,
    Keyword,
    PrimaryKeyAutoIncrease,
    PrimaryKeyUUID,
    Table,
    Vector,
    VectorIndex,
)

URL = "127.0.0.1"
# for local container development environment, use the host machine's IP
if environ.get("REMOTE_CONTAINERS", "") == "true" or environ.get("USER", "") == "envd":
    URL = "172.17.0.1"
TEST_POSTGRES = f"postgresql://postgres:postgres@{URL}:5432/"
DenseVector = Vector[128]


def gen_vector():
    rng = np.random.default_rng()
    return DenseVector(rng.random((128,), dtype=np.float32))


class Document(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    title: str = ""
    text: str
    updated_at: datetime = msgspec.field(default_factory=datetime.now)


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    doc_id: Annotated[int, ForeignKey[Document.uid]]
    text: str
    vector: DenseVector
    keyword: Keyword


class AnnotatedChunk(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    text: str
    vector: Annotated[DenseVector, VectorIndex(distance="cos", lists=2)]


class Sentence(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    vector: list[DenseVector]


@pytest.fixture(name="registry")
def fixture_registry(request):
    registry = VechordRegistry(request.node.name, TEST_POSTGRES)
    registry.register([Document, Chunk])
    yield registry
    logger.debug("clearing storage...")
    registry.clear_storage(drop_table=True)


@pytest.mark.db
def test_insert_select_remove(registry):
    docs = [Document(text="hello world"), Document(text="hello there")]
    for doc in docs:
        registry.insert(doc)

    # select all
    inserted = registry.select_by(Document.partial_init(), fields=["text"])
    assert len(inserted) == len(docs)
    assert inserted[0].text == "hello world"
    assert inserted[1].text == "hello there"

    # select with limit
    one = registry.select_by(Document.partial_init(), limit=1)
    assert len(one) == 1

    # select by id
    first = registry.select_by(Document.partial_init(uid=1))
    assert len(first) == 1
    assert first[0].text == "hello world"

    # remove by id
    registry.remove_by(Document.partial_init(uid=2))
    assert len(registry.select_by(Document.partial_init())) == 1


@pytest.mark.db
def test_annotated_index(registry):
    registry.register([AnnotatedChunk])
    num = 100
    topk = 5
    for text in (f"hello {i}" for i in range(num)):
        registry.insert(AnnotatedChunk(text=text, vector=gen_vector()))

    inserted = registry.select_by(AnnotatedChunk.partial_init(), fields=["text"])
    assert len(inserted) == num

    res = registry.search_by_vector(AnnotatedChunk, gen_vector(), topk=topk)
    assert len(res) == topk


@pytest.mark.db
def test_foreign_key(registry):
    docs = [
        Document(text="hello world"),
        Document(text="hello there"),
    ]
    chunks = [
        Chunk(doc_id=1, text="hello", keyword=Keyword("hello"), vector=gen_vector()),
        Chunk(doc_id=1, text="world", keyword=Keyword("world"), vector=gen_vector()),
    ]
    for record in docs + chunks:
        registry.insert(record)

    assert len(registry.select_by(Document.partial_init())) == len(docs)
    # remove the doc should also remove the related chunks
    registry.remove_by(Document.partial_init(uid=1))
    assert len(registry.select_by(Document.partial_init())) == 1
    assert len(registry.select_by(Chunk.partial_init())) == 0


@pytest.mark.db
def test_injection(registry):
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
    create_doc(text)
    create_chunk()

    docs = registry.select_by(Document.partial_init())
    assert len(docs) == 1

    chunks = registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(text.split())

    topk = 3
    # vector search
    vec_res = registry.search_by_vector(Chunk, gen_vector(), topk=topk)
    assert len(vec_res) == topk
    assert all(chunk.text in text for chunk in vec_res)
    # keyword search
    text_res = registry.search_by_keyword(Chunk, "vector", topk=topk)
    assert len(text_res) == 1


@pytest.mark.db
def test_multi_vec_maxsim(registry):
    registry.register([Sentence])

    @registry.inject(output=Sentence)
    def create_sentence(text: str) -> Sentence:
        return Sentence(
            text=text, vector=[gen_vector() for _ in range(len(text.split()))]
        )

    text = "the quick brown fox jumps over the lazy dog"
    num = 32
    for _ in range(num):
        create_sentence(text)
    sentence = registry.select_by(Sentence.partial_init())
    assert len(sentence) == num
    assert len(sentence[0].vector) == len(text.split())

    topk = 3
    for dim in range(1, 10):
        res = registry.search_by_multivec(
            Sentence, [gen_vector() for _ in range(dim)], topk=topk
        )
        assert len(res) == topk


@pytest.mark.db
def test_pipeline(registry):
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

    pipeline.run(correct)
    docs = registry.select_by(Document.partial_init())
    assert len(docs) == 1
    chunks = registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(correct.split())

    # break the transaction won't add new records
    with pytest.raises(ValueError):
        pipeline.run(error)

    docs = registry.select_by(Document.partial_init())
    assert len(docs) == 1
    chunks = registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(correct.split())
