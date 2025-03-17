import os
from abc import ABC, abstractmethod
from enum import Enum, auto

import msgspec
import numpy as np

from vechord.log import logger
from vechord.model import SparseEmbedding


class VecType(Enum):
    DENSE = auto()
    SPARSE = auto()
    KEYWORD = auto()


class BaseEmbedding(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def vectorize_chunk(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def vectorize_query(self, text: str) -> np.ndarray:
        return self.vectorize_chunk(text)

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def vec_type(self) -> VecType:
        raise NotImplementedError


class SpacyDenseEmbedding(BaseEmbedding):
    """Spacy Dense Embedding."""

    def __init__(self, model: str = "en_core_web_sm", dim: int = 96):
        import spacy

        self.nlp = spacy.load(model, enable=["tok2vec"])
        self.dim = dim
        self.model = model

    def name(self) -> str:
        return f"spacy_emb_{self.model}"

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self):
        return VecType.DENSE

    def vectorize_chunk(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        return doc.vector


class GeminiDenseEmbedding(BaseEmbedding):
    """Gemini Dense Embedding."""

    def __init__(self, model: str = "models/text-embedding-004", dim: int = 768):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("env GEMINI_API_KEY not set")

        import google.generativeai as genai

        self.client = genai.embed_content
        self.limit = 10000
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"gemini_emb_{self.model}_{self.dim}"

    def vectorize_chunk(self, text: str) -> np.ndarray:
        res = self.client(
            content=text[: self.limit], model=self.model, output_dimensionality=self.dim
        )
        return np.array(res["embedding"])


class OpenAIDenseEmbedding(BaseEmbedding):
    """OpenAI Dense Embedding."""

    def __init__(self, model: str = "text-embedding-3-large", dim: int = 3072):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("env OPENAI_API_KEY not set")

        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"openai_emb_{self.model}_{self.dim}"

    def vectorize_chunk(self, text: str) -> np.ndarray:
        return np.array(
            self.client.embeddings.create(
                model=self.model, input=text, dimensions=self.dim
            )
            .data[0]
            .embedding
        )


class SpladePPSparseEmbedding(BaseEmbedding):
    def __init__(self, url: str, dim: int = 30522, timeout_sec: int = 10):
        import httpx

        self.dim = dim
        self.client = httpx.Client(base_url=url, timeout=timeout_sec)
        self.decoder = msgspec.json.Decoder(type=[SparseEmbedding])

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.SPARSE

    def name(self) -> str:
        return f"spladepp_emb_{self.dim}"

    def vectorize_chunk(self, text: str | list[str]) -> list[SparseEmbedding]:
        resp = self.client.post("/inference", json=text)
        if resp.is_error:
            logger.info(
                "failed to call Splade++ sparse emb: [%d] %s",
                resp.status_code,
                resp.content,
            )
            resp.raise_for_status()

        sparse = self.decoder.decode(resp.content)
        return sparse
