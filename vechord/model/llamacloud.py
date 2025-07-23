from enum import Enum
from uuid import UUID

import msgspec

from vechord.typing import Self


# https://docs.cloud.llamaindex.ai/llamaparse/features/supported_document_types
# Currently only
class LlamaCloudMimeType(str, Enum):
    # Documents
    PDF = "application/pdf"
    EPUB = "application/epub+zip"
    RTF = "application/rtf"
    XML = "application/xml"
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TXT = "text/plain"
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLS = "application/vnd.ms-excel"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    CSV = "text/csv"
    JSON = "application/json"

    # Images
    JPG = "image/jpeg"
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    BMP = "image/bmp"
    SVG = "image/svg+xml"
    TIFF = "image/tiff"
    WEBP = "image/webp"

    # Audio/Video
    MP3 = "audio/mpeg"
    MP4 = "video/mp4"
    MPEG = "video/mpeg"
    MPGA = "audio/mpeg"
    M4A = "audio/mp4"
    WAV = "audio/wav"
    WEBM = "video/webm"


class LlamaCloudJobStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    CANCELLED = "CANCELLED"


class LlamaCloudParseRequest(msgspec.Struct, kw_only=True):
    filename: str
    content: bytes
    mime_type: LlamaCloudMimeType = LlamaCloudMimeType.PDF

    @classmethod
    def from_image(cls, img: bytes, mime_type: LlamaCloudMimeType) -> Self:
        # FIXME: Add real filename
        return cls(filename="image.jpg", content=img, mime_type=mime_type)

    @classmethod
    def from_pdf(cls, pdf: bytes, mime_type: LlamaCloudMimeType) -> Self:
        return cls(filename="document.pdf", content=pdf, mime_type=mime_type)


class LlamaCloudParseResponse(msgspec.Struct, kw_only=True):
    id: UUID
    status: LlamaCloudJobStatus
    error_code: int | None = None
    error_message: str | None = None
