from datetime import datetime

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
    updated_at: datetime


class Chunk(msgspec.Struct, kw_only=True):
    text: str
    vector: ndarray
    entities: list[Entity] = []
