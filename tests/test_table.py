from datetime import datetime
from os import environ
from typing import Annotated

import msgspec
import numpy as np
import pytest

from vechord.log import logger
from vechord.registry import VechordRegistry
from vechord.spec import ForeignKey, PrimaryKeyAutoIncrease, Table, Vector

URL = "127.0.0.1"
# for local container development environment, use the host machine's IP
if environ.get("REMOTE_CONTAINERS", "") == "true" or environ.get("USER", "") == "envd":
    URL = "172.17.0.1"
TEST_POSTGRES = f"postgresql://postgres:postgres@{URL}:5432/"
DenseVector = Vector[128]


def gen_vector():
    rng = np.random.default_rng()
    return rng.random((128,), dtype=np.float32)


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


@pytest.fixture(name="registry")
def fixture_registry(request):
    registry = VechordRegistry(request.node.name, TEST_POSTGRES)
    registry.register([Document, Chunk])
    yield registry
    logger.debug("clearing storage...")
    registry.clear_storage(drop_table=True)
    registry.pipeline.clear()


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

    # select by id
    first = registry.select_by(Document.partial_init(uid=1))
    assert len(first) == 1
    assert first[0].text == "hello world"

    # remove by id
    registry.remove_by(Document.partial_init(uid=2))
    assert len(registry.select_by(Document.partial_init())) == 1


@pytest.mark.db
def test_foreign_key(registry):
    docs = [
        Document(text="hello world"),
        Document(text="hello there"),
    ]
    chunks = [
        Chunk(doc_id=1, text="hello", vector=gen_vector()),
        Chunk(doc_id=1, text="world", vector=gen_vector()),
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
        return [Chunk(doc_id=uid, text=t, vector=gen_vector()) for t in text.split()]

    text = "hello world what happened to vector search"
    create_doc(text)
    create_chunk()

    docs = registry.select_by(Document.partial_init())
    assert len(docs) == 1

    chunks = registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(text.split())

    # test search
    topk = 3
    res = registry.search(Chunk, gen_vector(), topk=topk)
    assert len(res) == topk
    assert all(chunk.text in text for chunk in res)


@pytest.mark.db
def test_pipeline(registry):
    @registry.inject(output=Document)
    def create_doc(text: str) -> Document:
        return Document(text=text)

    @registry.inject(input=Document, output=Chunk)
    def create_chunk(uid: int, text: str) -> list[Chunk]:
        nums = [int(x) for x in text.split()]
        return [
            Chunk(doc_id=uid, text=f"num[{num}]", vector=gen_vector()) for num in nums
        ]

    correct = "1 2 3 4 5"
    error = "100 0.1 no no"
    registry.set_pipeline([create_doc, create_chunk])

    registry.run(correct)
    docs = registry.select_by(Document.partial_init())
    assert len(docs) == 1
    chunks = registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(correct.split())

    # break the transaction won't add new records
    with pytest.raises(ValueError):
        registry.run(error)

    docs = registry.select_by(Document.partial_init())
    assert len(docs) == 1
    chunks = registry.select_by(Chunk.partial_init())
    assert len(chunks) == len(correct.split())
