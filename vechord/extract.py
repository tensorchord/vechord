import unicodedata
from abc import ABC, abstractmethod
from html.parser import HTMLParser

import pypdfium2 as pdfium

from vechord.log import logger
from vechord.model import (
    Document,
    GeminiGenerateRequest,
    GeminiMimeType,
    LlamaCloudMimeType,
    LlamaCloudParseRequest,
)
from vechord.provider import GeminiGenerateProvider, LlamaCloudProvider


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


class GeminiExtractor(SimpleExtractor, GeminiGenerateProvider):
    """Extract text with Gemini model.

    Limits:
        - PDF: less than **20 MB** in size, max **3072x3072** with original aspect ratio
        - Image: less than **20 MB** in size, larger images are tiled into **768x768** tiles
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
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

    def name(self) -> str:
        return f"gemini_extractor_{self.model}"

    async def extract_image(self, img: bytes) -> str:
        """Extract text & caption from image."""
        resp = await self.query(
            GeminiGenerateRequest.from_prompt_with_data(
                self.jpeg_prompt, GeminiMimeType.JPEG, img
            )
        )
        return resp.get_text().strip()

    async def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF page by page."""
        resp = await self.query(
            GeminiGenerateRequest.from_prompt_with_data(
                self.pdf_prompt, GeminiMimeType.PDF, doc
            )
        )
        return resp.get_text().strip()


class LlamaParseExtractor(SimpleExtractor, LlamaCloudProvider):
    """Extract test with LlamaCloud Parse service.

    Limits:
        - Maximum run time for jobs : 30 minutes. If your job takes more than **30 minutes** to process, a TIMEOUT error will be raised.
        - Maximum size of files: **300Mb**.
        - Maximum image extracted / OCR per page: **35 images**. If more images are present in a page, only the 35 biggest one are extracted / OCR.
        - Maximum amount of text extracted per page: **64Kb**. Content beyond the 64Kb mark is ignored.
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "llama_parse_extractor"

    async def extract_image(self, img: bytes) -> str:
        """Extract text from image using LlamaCloud Parse service."""
        req = LlamaCloudParseRequest.from_image(img, LlamaCloudMimeType.JPEG)
        resp = await self.parse(req)
        text = await self.get_text(resp.id)
        if text is not None:
            return text.strip()
        return ""

    async def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF using LlamaCloud Parse service."""
        req = LlamaCloudParseRequest.from_pdf(doc, LlamaCloudMimeType.PDF)
        resp = await self.parse(req)
        text = await self.get_text(resp.id)
        if text is not None:
            return text.strip()
        return ""
