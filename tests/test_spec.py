from datetime import datetime
from typing import Annotated

import msgspec
import numpy as np
import pytest

from vechord.spec import ForeignKey, Keyword, PrimaryKeyAutoIncrease, Table, Vector


class Document(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    title: str
    text: str
    updated_at: datetime = msgspec.field(default_factory=datetime.now)


class Chunk(Table, kw_only=True):
    uid: PrimaryKeyAutoIncrease | None = None
    doc_id: Annotated[int, ForeignKey[Document.uid]]
    text: str
    vec: Vector[128]
    multivec: list[Vector[128]]
    keyword: Keyword


class Simple(Table):
    uid: int
    text: str


@pytest.mark.parametrize("table", [Document, Chunk, Simple])
def test_storage_cls_methods(table: type[Table]):
    assert table.name() == table.__name__.lower()
    assert "uid" in table.fields()

    t = table.partial_init()
    for field in t.fields():
        assert getattr(t, field) is msgspec.UNSET

    # UNSET won't appear in the `todict` result
    assert t.todict() == {}


def test_table_cls_methods():
    assert Document.primary_key() == "uid", Document
    assert Chunk.primary_key() == "uid", Chunk

    assert Document.vector_column() is None
    assert Chunk.vector_column() == "vec"
    assert Chunk.multivec_column() == "multivec"
    assert Chunk.keyword_column() == "keyword"

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
