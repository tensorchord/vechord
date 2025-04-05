from datetime import datetime
from typing import Annotated
from uuid import UUID

import msgspec
import numpy as np
import pytest

from vechord.spec import (
    DefaultDocument,
    ForeignKey,
    Keyword,
    MultiVectorIndex,
    PrimaryKeyAutoIncrease,
    PrimaryKeyUUID,
    Table,
    Vector,
    VectorDistance,
    VectorIndex,
    create_chunk_with_dim,
)


class Document(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    title: str
    text: str
    updated_at: datetime = msgspec.field(default_factory=datetime.now)


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    doc_id: Annotated[int, ForeignKey[Document.uid]]
    text: str
    vec: Vector[128]
    multivec: list[Vector[128]]
    keyword: Keyword


class Simple(Table):
    uid: int
    text: str


@pytest.mark.parametrize(
    "table", [Document, Chunk, Simple, DefaultDocument, create_chunk_with_dim(128)]
)
def test_storage_cls_methods(table: type[Table]):
    assert table.name() == table.__name__.lower()
    assert "uid" in table.fields()

    t = table.partial_init()
    for field in t.fields():
        assert getattr(t, field) is msgspec.UNSET

    # UNSET won't appear in the `todict` result
    assert t.todict() == {}


def test_table_todict():
    doc = Document(title="hello world", text="hello there")
    values = doc.todict()
    assert values["title"] == "hello world"
    assert values["text"] == "hello there"
    assert values["updated_at"] is not None, values
    assert "uid" not in values, values

    chunk = Chunk(
        doc_id=1,
        text="hello",
        vec=Vector[128]([0.1] * 128),
        multivec=[],
        keyword=Keyword("hello"),
    )
    values = chunk.todict()
    assert values["doc_id"] == 1
    assert values["text"] == "hello"
    assert np.isclose(values["vec"], np.array([0.1] * 128)).all()
    assert values["multivec"] == []
    assert values["keyword"] == "hello"
    assert values["uid"], values


def test_default_chunk():
    dim = 1024
    chunk_cls = create_chunk_with_dim(dim)
    DenseVector = Vector[dim]
    text = "hello world"
    chunk = chunk_cls(
        doc_id=0, text=text, vec=DenseVector(np.ones(dim)), keyword=Keyword(text)
    )

    assert isinstance(chunk.uid, UUID)
    values = chunk.todict()
    assert values["doc_id"] == 0


def test_table_cls_methods():
    assert Document.primary_key() == "uid", Document
    assert Chunk.primary_key() == "uid", Chunk

    assert Document.vector_column() is None
    assert Chunk.vector_column().name == "vec"
    assert Chunk.multivec_column().name == "multivec"
    assert Chunk.keyword_column().name == "keyword"

    def find_schema_by_name(schema, name):
        for n, t in schema:
            if n == name:
                return t

    assert "GENERATED ALWAYS AS IDENTITY" in find_schema_by_name(
        Document.table_schema(), "uid"
    )
    assert "VECTOR(128)" in find_schema_by_name(Chunk.table_schema(), "vec")
    assert (
        "REFERENCES {namespace}_document(uid) ON DELETE CASCADE"
        in find_schema_by_name(Chunk.table_schema(), "doc_id")
    )


def test_vector_type():
    Dense = Vector[128]

    # test the dim
    with pytest.raises(ValueError):
        Dense([0.1] * 100)

    with pytest.raises(ValueError):
        Dense(np.random.rand(123))

    assert np.equal(Dense(np.ones(128)), Dense([1.0] * 128)).all()


def test_index():
    with pytest.raises(msgspec.ValidationError):
        VectorIndex(distance="bug")

    with pytest.raises(msgspec.ValidationError):
        VectorIndex(lists="bug")

    lists = 128
    vec_idx = VectorIndex(distance="l2", lists=lists)
    assert vec_idx.distance is VectorDistance.L2
    assert vec_idx.lists == lists
    assert vec_idx.op_symbol == "<->"
    assert str(vec_idx.lists) in vec_idx.config()

    multivec_idx = MultiVectorIndex(lists=1)
    assert multivec_idx.lists == 1
    assert MultiVectorIndex().config() == "build.internal.lists = []"


def test_annotated_index():
    lists = 8

    class Sentence(Table, kw_only=True):
        uid: PrimaryKeyAutoIncrease | None = None
        text: str
        vec: Annotated[Vector[128], VectorIndex(distance="cos", lists=lists)]
        vecs: Annotated[list[Vector[128]], MultiVectorIndex(lists=lists)]

    vec_col = Sentence.vector_column()
    assert vec_col
    assert vec_col.name == "vec"
    assert vec_col.index.op_name == "vector_cosine_ops"
    assert vec_col.index.lists == lists

    multivec_col = Sentence.multivec_column()
    assert multivec_col
    assert multivec_col.name == "vecs"
    assert multivec_col.index.lists == lists
