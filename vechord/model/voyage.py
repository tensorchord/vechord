import base64
import re
from typing import Literal, Optional

import msgspec
import numpy as np

from vechord.errors import RequestError, UnexpectedResponseError
from vechord.typing import Self


def pascal_to_snake(s: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


class VoyageEmbedding(msgspec.Struct, kw_only=True):
    embedding: list[float] | bytes  # msgspec will handle the base64 encoding/decoding
    index: int


class VoyageEmbeddingResponse(msgspec.Struct, kw_only=True):
    data: list[VoyageEmbedding]

    def get_emb(self) -> np.ndarray:
        """Get the first embedding as a numpy array."""
        if not self.data or not self.data[0].embedding:
            raise UnexpectedResponseError("empty embedding data")
        emb = self.data[0].embedding
        if isinstance(emb, list):
            return np.array(emb, dtype=np.float32)
        return np.frombuffer(emb, dtype=np.float32)


VOYAGE_INPUT_TYPE = Literal["query", "document"]


class VoyageEmbeddingRequest(msgspec.Struct, kw_only=True):
    model: str
    input_text: str | list[str] = msgspec.field(name="input")
    input_type: VOYAGE_INPUT_TYPE = "document"
    truncation: bool = True
    output_dimension: int
    output_dtype: Literal["float", "int8", "uint8", "binary", "ubinary"] = "float"
    encoding_format: Optional[Literal["base64"]] = "base64"

    @classmethod
    def from_text(
        cls, text: str, input_type: VOYAGE_INPUT_TYPE, model: str, dim: int
    ) -> Self:
        return cls(
            model=model,
            input_text=text,
            input_type=input_type,
            output_dimension=dim,
        )


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
    input_type: VOYAGE_INPUT_TYPE = "document"
    truncation: bool = True
    encoding_format: Optional[Literal["base64"]] = "base64"

    @classmethod
    def build(
        cls,
        text: Optional[str],
        image: Optional[bytes],
        image_url: Optional[str],
        model: str,
        input_type: VOYAGE_INPUT_TYPE,
    ) -> Self:
        if not (text or image_url or image):
            raise RequestError(
                "At least one of text, image_url, or image must be provided."
            )
        contents = []
        if text:
            contents.append(Text(text=text))
        if image_url:
            contents.append(ImageURL(image_url=image_url))
        if image:
            contents.append(
                ImageBase64(
                    image_base64=f"data:image/jpeg;base64,{base64.b64encode(image).decode('utf-8')}",
                )
            )
        return cls(
            model=model,
            inputs=[MultiModalInput(content=contents)],
            input_type=input_type,
        )
