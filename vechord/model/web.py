from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

import msgspec


class ResourceRequest(msgspec.Struct, kw_only=True):
    kind: Literal[
        "ocr",
        "chunk",
        "text-emb",
        "multimodal-emb",
        "rerank",
        "index",
        "search",
        "evaluate",
    ]
    provider: str
    args: dict[str, Any] = msgspec.field(default_factory=dict)


class InputType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image/jpeg"


class RunRequest(msgspec.Struct, kw_only=True, frozen=True):
    """Request to run a dynamic pipeline.

    `steps` can be omitted if the it's already defined in the pipeline.
    """

    name: str
    data: bytes
    input_type: InputType = InputType.TEXT
    steps: list[ResourceRequest] = msgspec.field(default_factory=list)


class RunIngestAck(msgspec.Struct, kw_only=True, frozen=True, tag="ingest"):
    """Acknowledgment of an index request."""

    name: str
    msg: str
    uid: UUID


class SearchResponse(msgspec.Struct, kw_only=True, omit_defaults=True):
    uid: UUID
    doc_id: UUID
    text: Optional[str] = None


class RunSearchResponse(msgspec.Struct, kw_only=True, omit_defaults=True, tag="search"):
    """Response to a search request.

    metrics:
    - MRR
    - precision@k
    - average precision@k
    """

    chunks: list[SearchResponse] = msgspec.field(default_factory=list)
    chunk_type: Optional[str] = None
    metrics: dict[str, float] = msgspec.field(default_factory=dict)

    def extend(self, chunks: list):
        if chunks and self.chunk_type is None:
            self.chunk_type = (
                chunks[0].text_type if hasattr(chunks[0], "text_type") else "text"
            )
        for chunk in chunks:
            self.chunks.append(
                SearchResponse(
                    uid=chunk.uid,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                )
            )

    def cleanup(self):
        if self.chunk_type and self.chunk_type != "text":
            for chunk in self.chunks:
                chunk.text = None

    def deduplicate(self):
        """Deduplicate chunks while maintain the order."""
        unique_chunks = {}
        for chunk in self.chunks:
            if chunk.uid not in unique_chunks:
                unique_chunks[chunk.uid] = chunk
        self.chunks = list(unique_chunks.values())

    def reorder(self, indices: list[int]):
        self.chunks = [self.chunks[i] for i in indices]
