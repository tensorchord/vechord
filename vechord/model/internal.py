from datetime import datetime
from typing import Annotated

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
    label: str
    description: str


class GraphRelation(msgspec.Struct, kw_only=True, frozen=True):
    """
    Attributes:
    - source: the source entity (text, label and description)
    - target: the target entity (text, label and description)
    - description: a brief description of the relation in the current context
    """

    source: GraphEntity
    target: GraphEntity
    description: str


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


class UMBRELAScore(msgspec.Struct, kw_only=True):
    """Score for UMBRELA evaluation.

    - 0: Not relevant
    - 1: Relevant but does not answer the query
    - 2: Answer the query but may be a bit unclear
    - 3: Dedicated to the query and contains the exact answer
    """

    score: Annotated[int, msgspec.Meta(ge=0, le=3)]
