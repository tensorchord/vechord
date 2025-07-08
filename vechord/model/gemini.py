from enum import Enum
from typing import Any, Literal, Optional

import msgspec


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
    def from_prompt_with_data(cls, prompt: str, mime_type: GeminiMimeType, data: bytes):
        return cls(
            contents=Part(
                [
                    ContentPart(text=prompt),
                    ContentPart(inline_data=InlineData(mime_type=mime_type, data=data)),
                ]
            )
        )

    @classmethod
    def from_prompt(cls, prompt: str):
        return cls(contents=Part(parts=[ContentPart(text=prompt)]))

    @classmethod
    def from_prompt_structure_response(cls, prompt: str, schema: dict[str, Any]):
        return cls(
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


if __name__ == "__main__":
    req = GeminiGenerateRequest.from_prompt_with_data(
        "prompt", GeminiMimeType.PDF, b"PDF content"
    )
    b = msgspec.json.encode(req)
    print(b)
    print(msgspec.json.decode(b, type=GeminiGenerateRequest))

    struct = GeminiGenerateRequest.from_prompt_structure_response(
        "prompt", {"type": "object", "properties": {"key": {"type": "string"}}}
    )
    print(msgspec.json.encode(struct))
