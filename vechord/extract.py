import base64
import os
import unicodedata
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from io import BytesIO

import pypdfium2 as pdfium

from vechord.log import logger
from vechord.model import Document


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
    def extract_pdf(self, doc: bytes) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_html(self, text: str) -> str:
        raise NotImplementedError

    def extract(self, doc: Document) -> str:
        if doc.ext == ".txt":
            text = doc.data.decode("utf-8")
        elif doc.ext == ".pdf":
            text = self.extract_pdf(doc.data)
        elif doc.ext == ".html":
            text = self.extract_html(doc.data.decode("utf-8"))
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

    def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF using pypdfium2."""
        pdf = pdfium.PdfDocument(doc)
        text = []
        for page in pdf:
            text.append(page.get_textpage().get_text_bounded())

        return "\n".join(text)

    def extract_html(self, text: str) -> str:
        """Extract text from HTML.

        Args:
            text: HTML text.
        """
        parser = BaseHTMLParser()
        parser.feed(text)
        return "\n".join(t for t in parser.content if t)


class GeminiExtractor(SimpleExtractor):
    """Extract text with Gemini model."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("env GEMINI_API_KEY not set")

        import google.generativeai as genai

        self.model = genai.GenerativeModel(model)
        self.prompt = (
            "Extract all the text from the following document and return it exactly as"
            " it appears, without any modifications, summarization, or interpretation"
        )

    def name(self) -> str:
        return f"gemini_extractor_{self.model.model_name}"

    def extract_pdf(self, doc: bytes) -> str:
        """Extract text from PDF page by page."""
        pdf = pdfium.PdfDocument(doc)
        text = []
        for page in pdf:
            img = page.render(scale=2).to_pil()  # make the text clearer
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            response = self.model.generate_content(
                [
                    {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_bytes.getvalue()).decode("utf-8"),
                    },
                    self.prompt,
                ]
            )
            text.append(response.text)

        return "\n".join(text)
