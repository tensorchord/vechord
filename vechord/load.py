import hashlib
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from vechord.log import logger
from vechord.model import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class LocalLoader(BaseLoader):
    """Load documents from local file system."""

    def __init__(self, path: str, include: list[str] | None = None):
        self.path = Path(path)
        self.include = set(ext.lower() for ext in include or [".txt"])

    def name(self) -> str:
        return "local"

    def load(self) -> list[Document]:
        res = []
        for root, _dirs, files in os.walk(self.path):
            for file in files:
                filepath = Path(root) / file
                ext = filepath.suffix.lower()
                if ext not in self.include:
                    logger.debug("exclude file %s", file)
                    continue
                data = filepath.read_bytes()
                res.append(
                    Document(
                        path=str(filepath),
                        data=data,
                        ext=ext,
                        digest=hashlib.sha256(data).hexdigest(),
                        updated_at=datetime.fromtimestamp(filepath.stat().st_mtime),
                        source=self.name(),
                    )
                )
        return res


class S3Loader(BaseLoader):
    def __init__(self, bucket: str, prefix: str, include: list[str] | None = None):
        self.bucket = bucket
        self.prefix = prefix
        self.include = set(ext.lower() for ext in include or [".txt"])

    def name(self) -> str:
        return f"s3_{self.bucket}_{self.prefix}"

    def load(self) -> list[Document]:
        # TODO: implement S3 loader
        raise NotImplementedError
