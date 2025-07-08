from enum import Enum
from typing import Any, Literal
from uuid import UUID

import msgspec


class ResourceRequest(msgspec.Struct, kw_only=True):
    kind: Literal[
        "ocr", "chunk", "text-emb", "multimodal-emb", "rerank", "index", "search"
    ]
    provider: str
    args: dict[str, Any] = msgspec.field(default_factory=dict)


class InputType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image/jpeg"


class RunRequest(msgspec.Struct, kw_only=True, frozen=True):
    """Request to run a dynamic pipeline.

    Possible `data` types:
        - text
        - PDF
        - image/jpeg
    """

    name: str
    data: bytes
    input_type: InputType = InputType.TEXT
    steps: list[ResourceRequest]


class RunAck(msgspec.Struct, kw_only=True, frozen=True):
    name: str
    msg: str
    uid: UUID
