from datetime import datetime
from typing import Optional

import msgspec
from numpy import ndarray


class Entity(msgspec.Struct, kw_only=True, frozen=True):
    text: str
    label: str


class Document(msgspec.Struct, kw_only=True):
    ext: str
    path: str
    digest: str
    data: bytes
    updated_at: datetime = msgspec.field(default_factory=datetime.now)
    source: str = ""


class SparseEmbedding(msgspec.Struct, kw_only=True, frozen=True):
    dim: int
    indices: list[int]
    values: list[float]


class Keywords(msgspec.Struct, kw_only=True):
    words: list[str]
    weights: list[float]


class Chunk(msgspec.Struct, kw_only=True):
    text: str
    seq_id: int = 0
    vector: Optional[ndarray] = None
    sparse: Optional[SparseEmbedding] = None
    keywords: Optional[Keywords] = None
    entities: list[Entity] = []


class RetrievedChunk(msgspec.Struct, kw_only=True):
    uid: str
    text: str
    score: float
