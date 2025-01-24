import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path

from vechord.log import logger
from vechord.model import File


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[File]:
        raise NotImplementedError


class LocalLoader(BaseLoader):
    def __init__(self, path: str, include: list[str] | None = None):
        self.path = Path(path)
        self.include = set(ext.lower() for ext in include or [".txt"])

    def load(self) -> list[File]:
        res = []
        for root, _dirs, files in os.walk(self.path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in self.include:
                    logger.debug("exclude file %s", file)
                    continue
                data = (Path(root) / file).read_bytes()
                res.append(
                    File(
                        path=str(Path(root) / file),
                        data=data,
                        ext=ext,
                        digest=hashlib.sha256(data).hexdigest(),
                    )
                )
        return res
