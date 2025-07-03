import base64
import os
import unicodedata
from abc import ABC, abstractmethod
from html.parser import HTMLParser

import httpx
import pypdfium2 as pdfium

from vechord.log import logger
from vechord.model import Document
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
    """Extract text with Gemini model."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("env GEMINI_API_KEY not set")

        self.model = model
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self.prompt = (
            "Extract the main content from the PDF document. Ensure to exclude any "
            "metadata, headers, footers, or any other non-essential information. "
            "Return the extracted content as it appears in the document, without "
            "any additional modification, summarization or interpretation."
        )
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, read=120.0),
            headers={"Content-Type": "application/json"},
            transport=RateLimitTransport(max_per_second=GEMINI_GENERATE_RPS),
        )

    def name(self) -> str:
        return f"gemini_extractor_{self.model}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()

    async def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF page by page."""
        content = base64.b64encode(doc).decode("utf-8")
        response = await self.client.post(
            self.url,
            params={"key": self.api_key},
            json={
                "contents": {
                    "parts": [
                        {"text": self.prompt},
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": content,
                            }
                        },
                    ]
                }
            },
        )

        if response.is_error:
            raise RuntimeError(f"Failed to extract PDF with Gemini: {response.text}")

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
