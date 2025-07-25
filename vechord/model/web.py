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
    steps: list[ResourceRequest] = msgspec.field(default_factory=list)


class RunAck(msgspec.Struct, kw_only=True, frozen=True):
    """Acknowledgment of an index request."""

    name: str
    msg: str
    uid: UUID


class SearchResponse(msgspec.Struct, kw_only=True):
    uid: UUID
    doc_id: UUID
    text: str


class RunResponse(msgspec.Struct, kw_only=True, omit_defaults=True):
    """Response to a search request.

    metrics:
    - MRR
    - precision@k
    - average precision@k
    """

    chunks: list[SearchResponse] = msgspec.field(default_factory=list)
    metrics: dict[str, float] = msgspec.field(default_factory=dict)

    def extend(self, chunks: list):
        for chunk in chunks:
            self.chunks.append(
                SearchResponse(
                    uid=chunk.uid,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                )
            )

    def reorder(self, indices: list[int]):
        self.chunks = [self.chunks[i] for i in indices]
