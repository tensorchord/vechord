from typing import Literal, Optional

import msgspec
import numpy as np

from vechord.errors import UnexpectedResponseError
from vechord.typing import Self

# https://jina.ai/api-dashboard/embedding
JinaEmebeddingType = Literal[
    "retrieval.query",
    "retrieval.passage",
    "code.query",
    "code.passage",
    "text-matching",
]


class JinaInput(msgspec.Struct, kw_only=True, omit_defaults=True):
    text: Optional[str] = None
    image: Optional[str | bytes] = None  # URL or base64 encoded image
    pdf: Optional[str] = None  # URL or base64 encoded PDF


class JinaEmbeddingRequest(msgspec.Struct, kw_only=True, omit_defaults=True):
    model: Literal["jina-embeddings-v4", "jina-embeddings-v3"]
    dimensions: int = 2048
    truncate: bool
    task: JinaEmebeddingType
    embedding_type: Literal["binary", "ubinary", "base64", "float"] = "float"
    input_content: list[JinaInput] = msgspec.field(name="input")

    @classmethod
    def from_text(cls, text: str, task: JinaEmebeddingType) -> Self:
        return JinaEmbeddingRequest(
            model="jina-embeddings-v4",
            truncate=True,
            task=task,
            embedding_type="base64",
            input_content=[JinaInput(text=text)],
        )


class EmbeddingObject(msgspec.Struct, kw_only=True, omit_defaults=True):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | bytes


class JinaEmbeddingResponse(msgspec.Struct, kw_only=True):
    data: list[EmbeddingObject]

    def get_emb(self) -> np.ndarray:
        """Get the first embedding as a numpy array."""
        if not self.data or not self.data[0].embedding:
            raise UnexpectedResponseError("empty embedding data")
        return np.frombuffer(self.data[0].embedding, dtype=np.float32)
