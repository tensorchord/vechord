import os
import unicodedata
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from vechord.log import logger

DEFAULT_EXTRACTOR: dict[str, Callable[[bytes], str]] = {}


def extractor_register(extension: str):
    def wrapper(func):
        if extension in DEFAULT_EXTRACTOR:
            logger.warning("overriding extractor for %s", extension)
        DEFAULT_EXTRACTOR[extension] = func
        return func

    return wrapper


@extractor_register(".pdf")
def extract_pdf(stream: bytes) -> str:
    reader = PdfReader(BytesIO(stream))
    return " ".join(
        unicodedata.normalize("NFKC", page.extract_text()) for page in reader.pages
    )


class DataLoader:
    def __init__(self, exclude: list[str] | None = None):
        self.exclude = set(ext.lower() for ext in exclude or [])

    def supported_extensions(self) -> list[str]:
        return DEFAULT_EXTRACTOR.keys()

    def local_files(self, path: Path):
        for root, _dirs, files in os.walk(path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.exclude:
                    logger.debug("exclude file %s", file)
                    continue
                elif ext not in DEFAULT_EXTRACTOR:
                    logger.debug("unknown file type: %s", file)
                    continue
                yield Path(root) / file
