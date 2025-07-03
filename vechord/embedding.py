import base64
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Literal, Optional

import httpx
import msgspec
import numpy as np

from vechord.log import logger
from vechord.model import (
    SparseEmbedding,
    VoyageEmbeddingRequest,
    VoyageEmbeddingResponse,
    VoyageMultiModalEmbeddingRequest,
)
from vechord.utils import GEMINI_EMBEDDING_RPS, VOYAGE_EMBEDDING_RPS, RateLimitTransport


class VecType(Enum):
    DENSE = auto()
    SPARSE = auto()
    KEYWORD = auto()


class BaseEmbedding(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def vectorize_chunk(self, text: str) -> np.ndarray:
        raise NotImplementedError

    async def vectorize_query(self, text: str) -> np.ndarray:
        return await self.vectorize_chunk(text)

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def vec_type(self) -> VecType:
        raise NotImplementedError


class SpacyDenseEmbedding(BaseEmbedding):
    """Spacy Dense Embedding.

    The default small model is unlikely to be useful for real applications.
    It's recommended to use the small model for development and testing
    purpose only.
    """

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

    async def vectorize_chunk(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        return doc.vector


class GeminiDenseEmbedding(BaseEmbedding):
    """Gemini Dense Embedding.

    Args:
        model: Gemini embedding model name
        dim: embedding dimension, could be [768, 1536, 3072]
    """

    def __init__(
        self,
        model: str = "gemini-embedding-exp-03-07",
        dim: Literal[768, 1536, 3072] = 3072,
    ):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("env GEMINI_API_KEY not set")

        # this limit is 8192 tokens
        self.limit = 10000
        self.model = model
        self.dim = dim
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent"
        )
        self.client = httpx.AsyncClient(
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=RateLimitTransport(max_per_second=GEMINI_EMBEDDING_RPS),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"gemini_emb_{self.model}_{self.dim}"

    async def vectorize(
        self, text: str, task: str = "SEMANTIC_SIMILARITY"
    ) -> np.ndarray:
        resp = await self.client.post(
            self.url,
            json={"taskType": task, "content": {"parts": [{"text": text}]}},
        )
        if resp.is_error:
            raise RuntimeError(
                f"failed to call Gemini emb: [{resp.status_code}] {resp.content}"
            )
        data = resp.json()
        return np.array(data["embedding"]["values"], dtype=np.float32)

    async def vectorize_chunk(self, text: str) -> np.ndarray:
        return await self.vectorize(text, task="RETRIEVAL_DOCUMENT")

    async def vectorize_query(self, text):
        return await self.vectorize(text, task="RETRIEVAL_QUERY")


class VoyageDenseEmbedding(BaseEmbedding):
    def __init__(
        self, model: str = "voyage-3.5", dim: Literal[256, 512, 1024, 2048] = 1024
    ):
        self.api_key = os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("env VOYAGE_API_KEY not set")

        self.model = model
        self.dim = dim
        self.url = "https://api.voyageai.com/v1/embeddings"
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=RateLimitTransport(max_per_second=VOYAGE_EMBEDDING_RPS),
        )
        self.decoder = msgspec.json.Decoder(type=VoyageEmbeddingResponse)
        self.encoder = msgspec.json.Encoder()

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()

    def name(self):
        return f"voyage_emb_{self.model}_{self.dim}"

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    async def vectorize(
        self, text: str, input_type: Literal["document", "query"] = "document"
    ) -> np.ndarray:
        resp = await self.client.post(
            url=self.url,
            content=self.encoder.encode(
                VoyageEmbeddingRequest(
                    model=self.model,
                    input_text=text,
                    input_type=input_type,
                    output_dimension=self.dim,
                )
            ),
        )
        if resp.is_error:
            raise RuntimeError(
                f"failed to call Voyage emb: [{resp.status_code}] {resp.content}"
            )
        body = self.decoder.decode(resp.content)
        emb = np.frombuffer(body.data[0].embedding, dtype=np.float32)
        return emb

    async def vectorize_chunk(self, text):
        return await self.vectorize(text, "document")

    async def vectorize_query(self, text):
        return await self.vectorize(text, "query")


class VoyageMultiModalEmbedding(VoyageDenseEmbedding):
    def __init__(self, model="voyage-multimodal-3", dim=1024):
        super().__init__(model, dim)
        self.url = "https://api.voyageai.com/v1/multimodalembeddings"

    def name(self):
        return f"voyage_multimodal_emb_{self.model}_{self.dim}"

    async def vectorize(self, text, input_type: Literal["document", "query"] = "query"):
        await self.vectorize_multimodal(text=text, input_type=input_type)

    async def vectorize_multimodal(
        self,
        image: Optional[bytes] = None,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
        input_type: Literal["query", "document"] = "document",
    ):
        if not (image or text or image_url):
            raise ValueError(
                "At least one of image, text, or image_url must be provided"
            )

        input_content = []
        if text:
            input_content.append({"type": "text", "text": text})
        if image:
            input_content.append(
                {
                    "type": "image_base64",
                    "image_base64": f"data:image/jpeg;base64,{base64.b64encode(image).decode('utf-8')}",
                }
            )
        if image_url:
            input_content.append({"type": "image_url", "image_url": image_url})

        resp = await self.client.post(
            url=self.url,
            content=self.encoder.encode(
                VoyageMultiModalEmbeddingRequest(
                    model=self.model,
                    inputs=[{"content": input_content}],
                    input_type=input_type,
                )
            ),
        )
        if resp.is_error:
            raise RuntimeError(
                f"failed to call Voyage multimodal emb: [{resp.status_code}] {resp.content}"
            )
        body = self.decoder.decode(resp.content)
        emb = np.frombuffer(body.data[0].embedding, dtype=np.float32)
        return emb

    async def vectorize_multimodal_chunk(
        self,
        image: Optional[bytes] = None,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        return await self.vectorize_multimodal(
            image=image, text=text, image_url=image_url, input_type="document"
        )

    async def vectorize_multimodal_query(
        self,
        image: Optional[bytes] = None,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        return await self.vectorize_multimodal(
            image=image, text=text, image_url=image_url, input_type="query"
        )


class OpenAIDenseEmbedding(BaseEmbedding):
    """OpenAI Dense Embedding."""

    def __init__(self, model: str = "text-embedding-3-large", dim: int = 3072):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("env OPENAI_API_KEY not set")

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"openai_emb_{self.model}_{self.dim}"

    async def vectorize_chunk(self, text: str) -> np.ndarray:
        resp = await self.client.embeddings.create(
            model=self.model, input=text, dimensions=self.dim
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)


class SpladePPSparseEmbedding:
    """Splade++ Sparse Embedding."""

    def __init__(self, url: str, dim: int = 30522, timeout_sec: int = 10):
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
