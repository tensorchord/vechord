from enum import Enum
from typing import Any, Literal, Optional

import msgspec
import numpy as np

from vechord.typing import Self


# https://ai.google.dev/gemini-api/docs/document-processing#technical-details
# https://ai.google.dev/gemini-api/docs/image-understanding#supported-formats
class GeminiMimeType(str, Enum):
    # documents
    PDF = "application/pdf"
    JavaScript = "application/x-javascript"
    Python = "application/x-python"
    HTML = "text/html"
    CSS = "text/css"
    Markdown = "text/markdown"
    CSV = "text/csv"
    XML = "text/xml"
    RTF = "text/rtf"
    # images
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    HEIC = "image/heic"
    HEIF = "image/heif"


class UMBRELAScore(msgspec.Struct, kw_only=True):
    """Score for UMBRELA evaluation.

    - 0: Not relevant
    - 1: Weakly relevant
    - 2: Relevant
    - 3: Highly relevant
    """

    score: int = Literal[0, 1, 2, 3]


class InlineData(msgspec.Struct, kw_only=True):
    mime_type: GeminiMimeType
    data: bytes


class ContentPart(msgspec.Struct, kw_only=True, omit_defaults=True):
    text: str | None = None
    inline_data: InlineData | None = None


class Part(msgspec.Struct):
    parts: list[ContentPart] = msgspec.field(default_factory=list)


class GenerationConfig(msgspec.Struct, kw_only=True):
    response_mime_type: Literal["application/json"] = "application/json"
    response_json_schema: dict[str, Any] | None = None


class GeminiGenerateRequest(msgspec.Struct, kw_only=True):
    contents: Part
    generation_config: Optional[GenerationConfig] = msgspec.field(
        default=None, name="generationConfig"
    )

    @classmethod
    def from_prompt_with_data(
        cls, prompt: str, mime_type: GeminiMimeType, data: bytes
    ) -> Self:
        return GeminiGenerateRequest(
            contents=Part(
                [
                    ContentPart(text=prompt),
                    ContentPart(inline_data=InlineData(mime_type=mime_type, data=data)),
                ]
            )
        )

    @classmethod
    def from_prompt(cls, prompt: str) -> Self:
        return GeminiGenerateRequest(contents=Part(parts=[ContentPart(text=prompt)]))

    @classmethod
    def from_prompt_structure_response(
        cls, prompt: str, schema: dict[str, Any]
    ) -> Self:
        return GeminiGenerateRequest(
            contents=Part(parts=[ContentPart(text=prompt)]),
            generation_config=GenerationConfig(response_json_schema=schema),
        )


class ContentResponse(msgspec.Struct, kw_only=True, omit_defaults=True):
    content: Part


class GeminiGenerateResponse(msgspec.Struct, kw_only=True):
    candidates: list[ContentResponse] = msgspec.field(default_factory=list)

    def get_text(self) -> str:
        """Get the text from the first candidate's content part."""
        if not self.candidates or not self.candidates[0].content.parts:
            return ""
        return self.candidates[0].content.parts[0].text or ""


# https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types
GeminiEmbeddingType = Literal[
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "CODE_RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
]


class GeminiEmbeddingRequest(msgspec.Struct, kw_only=True, omit_defaults=True):
    model: Optional[str] = None
    task_type: GeminiEmbeddingType = msgspec.field(
        default="SEMANTIC_SIMILARITY", name="taskType"
    )
    content: Part

    @classmethod
    def from_text_with_type(
        cls, text: str, task_type: GeminiEmbeddingType = "SEMANTIC_SIMILARITY"
    ) -> Self:
        return GeminiEmbeddingRequest(
            content=Part(parts=[ContentPart(text=text)]),
            task_type=task_type,
        )


class Embedding(msgspec.Struct):
    values: list[float]


class GeminiEmbeddingResponse(msgspec.Struct, kw_only=True):
    embedding: Embedding

    def get_emb(self) -> np.ndarray:
        """Get the first embedding as a numpy array."""
        return np.array(self.embedding.values, dtype=np.float32)
