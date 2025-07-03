import re
from typing import Literal, Optional

import msgspec


def pascal_to_snake(s: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


class VoyageEmbedding(msgspec.Struct, kw_only=True):
    embedding: list[float] | bytes  # msgspec will handle the base64 encoding/decoding
    index: int


class VoyageEmbeddingResponse(msgspec.Struct, kw_only=True):
    data: list[VoyageEmbedding]


class VoyageEmbeddingRequest(msgspec.Struct, kw_only=True):
    model: str
    input_text: str | list[str] = msgspec.field(name="input")
    input_type: Literal["query", "document"] = "document"
    truncation: bool = True
    output_dimension: int
    output_dtype: Literal["float", "int8", "uint8", "binary", "ubinary"] = "float"
    encoding_format: Optional[Literal["base64"]] = "base64"


class Text(msgspec.Struct, tag=pascal_to_snake):
    text: str


class ImageBase64(msgspec.Struct, tag=pascal_to_snake):
    image_base64: str


class ImageURL(msgspec.Struct, tag=pascal_to_snake):
    image_url: str


class MultiModalInput(msgspec.Struct, tag=pascal_to_snake):
    content: list[Text | ImageBase64 | ImageURL]


class VoyageMultiModalEmbeddingRequest(msgspec.Struct, kw_only=True):
    model: str
    inputs: list[MultiModalInput]
    input_type: Literal["query", "document"] = "document"
    truncation: bool = True
    encoding_format: Optional[Literal["base64"]] = "base64"
