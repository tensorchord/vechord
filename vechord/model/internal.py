from datetime import datetime

import msgspec


# `frozen=True` is required to make it hashable
class GraphEntity(msgspec.Struct, kw_only=True, frozen=True):
    """
    Attributes:
    - text: the named entity text
    - label: the named entity type
    - description: a brief description of the entity in the current context
    """

    text: str
    label: str = ""
    description: str = ""


class GraphRelation(msgspec.Struct, kw_only=True, frozen=True):
    """
    Attributes:
    - source: the source entity (text, label and description)
    - target: the target entity (text, label and description)
    - description: a brief description of the relation in the current context
    """

    source: GraphEntity
    target: GraphEntity
    description: str = ""


class Document(msgspec.Struct, kw_only=True):
    ext: str
    data: bytes
    path: str = ""
    digest: str = ""
    source: str = ""
    updated_at: datetime = msgspec.field(default_factory=datetime.now)


class SparseEmbedding(msgspec.Struct, kw_only=True, frozen=True):
    dim: int
    indices: list[int]
    values: list[float]


class RetrievedChunk(msgspec.Struct, kw_only=True):
    uid: str
    text: str
    score: float
