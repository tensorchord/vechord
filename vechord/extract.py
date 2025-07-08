import os
import unicodedata
from abc import ABC, abstractmethod
from html.parser import HTMLParser

import httpx
import msgspec
import pypdfium2 as pdfium

from vechord.log import logger
from vechord.model import (
    Document,
    GeminiGenerateRequest,
    GeminiGenerateResponse,
    GeminiMimeType,
)
from vechord.utils import GEMINI_GENERATE_RPS, RateLimitTransport


class BaseHTMLParser(HTMLParser):
    """A simple HTML parser to extract text content."""

    def __init__(self) -> None:
        super().__init__()
        self.content: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.content.append(data.strip())


class BaseExtractor(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def extract_pdf(self, doc: bytes) -> str:
        raise NotImplementedError

    @abstractmethod
    async def extract_html(self, text: str) -> str:
        raise NotImplementedError

    async def extract(self, doc: Document) -> str:
        if doc.ext == ".txt":
            text = doc.data.decode("utf-8")
        elif doc.ext == ".pdf":
            text = await self.extract_pdf(doc.data)
        elif doc.ext == ".html":
            text = await self.extract_html(doc.data.decode("utf-8"))
        else:
            logger.warning("unsupported file type '%s' for %s", doc.ext, doc.path)
            text = ""
        return unicodedata.normalize("NFKC", text)


class SimpleExtractor(BaseExtractor):
    """Local extractor for text files."""

    def __init__(self):
        pass

    def name(self) -> str:
        return "basic_extractor"

    async def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF using pypdfium2."""
        pdf = pdfium.PdfDocument(doc)
        text = []
        for page in pdf:
            text.append(page.get_textpage().get_text_bounded())

        return "\n".join(text)

    async def extract_html(self, text: str) -> str:
        """Extract text from HTML.

        Args:
            text: HTML text.
        """
        parser = BaseHTMLParser()
        parser.feed(text)
        return "\n".join(t for t in parser.content if t)


class GeminiExtractor(SimpleExtractor):
    """Extract text with Gemini model.

    Limits:
        - PDF: less than **20 MB** in size, max **3072x3072** with original aspect ratio
        - Image: less than **20 MB** in size, larger images are tiled into **768x768** tiles
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("env GEMINI_API_KEY not set")

        self.model = model
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self.pdf_prompt = (
            "Extract the main content from the PDF document. Ensure to exclude any "
            "metadata, headers, footers, or any other non-essential information. "
            "Return the extracted content as it appears in the document, without "
            "any additional modification, summarization or interpretation."
        )
        self.jpeg_prompt = (
            "Extract the visible text from the image, generate a concise caption "
            "describing the image's content or scene, return the text with caption."
        )
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, read=120.0),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            transport=RateLimitTransport(max_per_second=GEMINI_GENERATE_RPS),
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(GeminiGenerateResponse)

    def name(self) -> str:
        return f"gemini_extractor_{self.model}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()

    async def query(self, prompt: str, mime_type: GeminiMimeType, stream: bytes) -> str:
        response = await self.client.post(
            self.url,
            content=self.encoder.encode(
                GeminiGenerateRequest.from_prompt_with_data(prompt, mime_type, stream)
            ),
        )
        if response.is_error:
            raise RuntimeError(
                f"Failed to query Gemini [{response.status_code}]: {response.text}"
            )

        data = self.decoder.decode(response.content)
        return data.get_text().strip()

    async def extract_image(self, img: bytes) -> str:
        """Extract text & caption from image."""
        return await self.query(self.jpeg_prompt, GeminiMimeType.JPEG, img)

    async def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF page by page."""
        return await self.query(self.pdf_prompt, GeminiMimeType.PDF, doc)
