from datetime import datetime
from typing import Any, Literal

import msgspec


# `frozen=True` is required to make it hashable
class Entity(msgspec.Struct, kw_only=True, frozen=True):
    text: str
    label: str = ""
    description: str = ""


class Relation(msgspec.Struct, kw_only=True, frozen=True):
    source: Entity
    target: Entity
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


class Keywords(msgspec.Struct, kw_only=True):
    words: list[str]
    weights: list[float]


class RetrievedChunk(msgspec.Struct, kw_only=True):
    uid: str
    text: str
    score: float


class ResourceRequest(msgspec.Struct, kw_only=True):
    kind: Literal["ocr", "chunk", "embedding", "rerank", "index", "search"]
    provider: str
    args: dict[str, Any] = msgspec.field(default_factory=dict)


class RunRequest(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    data: bytes
    steps: list[ResourceRequest]
