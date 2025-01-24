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


class LocalLoader(BaseLoader):
    def __init__(self, path: str, include: list[str] | None = None):
        self.path = Path(path)
        self.include = set(ext.lower() for ext in include or [".txt"])

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
                    )
                )
        return res
