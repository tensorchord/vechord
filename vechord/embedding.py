import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Literal, Optional

import httpx
import msgspec
import numpy as np

from vechord.log import logger
from vechord.model import (
    GeminiEmbeddingRequest,
    JinaEmbeddingRequest,
    SparseEmbedding,
    VoyageEmbeddingRequest,
    VoyageMultiModalEmbeddingRequest,
)
from vechord.provider import (
    GeminiEmbeddingProvider,
    JinaEmbeddingProvider,
    VoyageEmbeddingProvider,
)


class VecType(Enum):
    DENSE = auto()
    SPARSE = auto()
    KEYWORD = auto()


class BaseEmbedding(ABC):
    SUPPORT_MULTI_MODAL = False

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def vec_type(self) -> VecType:
        raise NotImplementedError

    def verify(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ):
        if not (text or image or image_url):
            raise ValueError(
                "At least one of `text`, `image`, or `image_url` must be provided."
            )
        if not self.SUPPORT_MULTI_MODAL and (image or image_url):
            raise ValueError(
                "This embedding model does not support multimodal input. Please use a text input."
            )

    @abstractmethod
    async def vectorize_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    async def vectorize_query(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        return await self.vectorize_chunk(text=text, image=image, image_url=image_url)


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

    async def vectorize_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
        doc = self.nlp(text)
        return doc.vector


class GeminiDenseEmbedding(BaseEmbedding, GeminiEmbeddingProvider):
    """Gemini Dense Embedding. (limit to **8,192** tokens)

    Args:
        model: Gemini embedding model name
        dim: embedding dimension, could be [768, 1536, 3072]
    """

    def __init__(
        self,
        model: str = "gemini-embedding-exp-03-07",
        dim: Literal[768, 1536, 3072] = 3072,
    ):
        super().__init__(model, dim)

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"gemini_emb_{self.model}_{self.dim}"

    async def vectorize_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            GeminiEmbeddingRequest.from_text_with_type(text, "RETRIEVAL_DOCUMENT")
        )
        return resp.get_emb()

    async def vectorize_query(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ):
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            GeminiEmbeddingRequest.from_text_with_type(text, "RETRIEVAL_QUERY")
        )
        return resp.get_emb()


class JinaDenseEmbedding(BaseEmbedding, JinaEmbeddingProvider):
    """Jina Dense Embedding. (limit to **32,768** tokens for v4, **8,192** for v3)

    Args:
        model: Jina embedding model name, could be "jina-embeddings-v4" or "jina-embeddings-v3"
        dim: embedding dimension, up to 2048
    """

    def __init__(self, model: str = "jina-embeddings-v4", dim: int = 2048):
        super().__init__(model, dim)

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"jina_emb_{self.model}_{self.dim}"

    async def vectorize_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            JinaEmbeddingRequest.from_text(text, "retrieval.passage", self.model)
        )
        return resp.get_emb()

    async def vectorize_query(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            JinaEmbeddingRequest.from_text(text, "retrieval.query", self.model)
        )
        return resp.get_emb()


class JinaMultiModalEmbedding(BaseEmbedding, JinaEmbeddingProvider):
    """Jina MultiModal Dense Embedding, limit to **32,768** tokens for v4.

    Args:
        model: Jina embedding model name, could be "jina-embeddings-v4"
        dim: embedding dimension, up to 2048
    """

    SUPPORT_MULTI_MODAL = True

    def __init__(self, model="jina-embeddings-v4", dim=2048):
        super().__init__(model, dim)

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"jina_emb_{self.model}_{self.dim}"

    async def vectorize_multimodal_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
        req = await self.query(
            JinaEmbeddingRequest.from_text_image(
                text=text,
                image=image,
                image_url=image_url,
                task="retrieval.passage",
                model=self.model,
            )
        )
        return req.get_emb()

    async def vectorize_multimodal_query(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
        req = await self.query(
            JinaEmbeddingRequest.from_text_image(
                text=text,
                image=image,
                image_url=image_url,
                task="retrieval.query",
                model=self.model,
            )
        )
        return req.get_emb()


class VoyageDenseEmbedding(BaseEmbedding, VoyageEmbeddingProvider):
    def __init__(
        self, model: str = "voyage-3.5", dim: Literal[256, 512, 1024, 2048] = 1024
    ):
        super().__init__(model, dim)

    def name(self):
        return f"voyage_emb_{self.model}_{self.dim}"

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    async def vectorize_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ):
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            VoyageEmbeddingRequest.from_text(
                text=text, input_type="document", model=self.model, dim=self.dim
            )
        )
        return resp.get_emb()

    async def vectorize_query(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ):
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            VoyageEmbeddingRequest.from_text(
                text=text, input_type="query", model=self.model, dim=self.dim
            )
        )
        return resp.get_emb()


class VoyageMultiModalEmbedding(BaseEmbedding, VoyageEmbeddingProvider):
    """Voyage Multimodal Embedding.

    Accepts text, image (as bytes), or image_url.

    Limits:
        - image: less than **16 million pixels** or **20 MB** in size
    """

    SUPPORT_MULTI_MODAL = True

    def __init__(self, model="voyage-multimodal-3", dim=1024):
        super().__init__(model, dim)
        self.url = "https://api.voyageai.com/v1/multimodalembeddings"

    def name(self):
        return f"voyage_multimodal_emb_{self.model}_{self.dim}"

    def get_dim(self):
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    async def vectorize_multimodal_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ):
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            VoyageMultiModalEmbeddingRequest.build(
                text=text,
                image=image,
                image_url=image_url,
                input_type="document",
                model=self.model,
            )
        )
        return resp.get_emb()

    async def vectorize_multimodal_query(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ):
        self.verify(text=text, image=image, image_url=image_url)
        resp = await self.query(
            VoyageMultiModalEmbeddingRequest.build(
                text=text,
                image=image,
                image_url=image_url,
                input_type="query",
                model=self.model,
            )
        )
        return resp.get_emb()


class OpenAIDenseEmbedding(BaseEmbedding):
    """OpenAI Dense Embedding."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dim: int = 3072,
        base_url: Optional[str] = None,
    ):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("env OPENAI_API_KEY not set")

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vec_type(self) -> VecType:
        return VecType.DENSE

    def name(self) -> str:
        return f"openai_emb_{self.model}_{self.dim}"

    async def vectorize_chunk(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> np.ndarray:
        self.verify(text=text, image=image, image_url=image_url)
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
