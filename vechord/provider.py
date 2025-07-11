from os import environ
from typing import Literal

import httpx
import msgspec

from vechord.errors import APIKeyUnsetError, HTTPCallError
from vechord.model import (
    GeminiEmbeddingRequest,
    GeminiEmbeddingResponse,
    GeminiGenerateRequest,
    GeminiGenerateResponse,
    JinaEmbeddingRequest,
    JinaEmbeddingResponse,
)
from vechord.utils import (
    GEMINI_EMBEDDING_RPS,
    GEMINI_GENERATE_RPS,
    JINA_EMBEDDING_RPS,
    RateLimitTransport,
)


class BaseProvider:
    PROVIDER_NAME = "UNKNOWN"

    def __init__(self, model: str):
        self.model = model
        api_key_name = f"{self.PROVIDER_NAME.upper()}_API_KEY"
        self.api_key = environ.get(api_key_name)
        if not self.api_key:
            raise APIKeyUnsetError(api_key_name)

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()


class GeminiGenerateProvider(BaseProvider):
    """Gemini Generate Provider."""

    PROVIDER_NAME = "GEMINI"

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
            transport=RateLimitTransport(max_per_second=GEMINI_GENERATE_RPS),
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(GeminiGenerateResponse)
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    async def query(self, req: GeminiGenerateRequest) -> GeminiGenerateResponse:
        """Query the Gemini model with a request."""
        response = await self.client.post(self.url, content=self.encoder.encode(req))
        if response.is_error:
            raise HTTPCallError(
                "Failed to query Gemini generate", response.status_code, response.text
            )
        return self.decoder.decode(response.content)


class GeminiEmbeddingProvider(BaseProvider):
    """Gemini Embedding Provider."""

    PROVIDER_NAME = "GEMINI"

    def __init__(
        self,
        model: str = "gemini-embedding-exp-03-07",
        dim: Literal[768, 1536, 3072] = 3072,
    ):
        super().__init__(model)
        self.dim = dim
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
            transport=RateLimitTransport(max_per_second=GEMINI_EMBEDDING_RPS),
        )
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent"
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(GeminiEmbeddingResponse)

    async def query(self, req: GeminiEmbeddingRequest) -> GeminiEmbeddingResponse:
        """Query the Gemini embedding model with a request."""
        response = await self.client.post(self.url, content=self.encoder.encode(req))
        if response.is_error:
            raise HTTPCallError(
                "Failed to query Gemini embedding", response.status_code, response.text
            )
        return self.decoder.decode(response.content)


class JinaEmbeddingProvider(BaseProvider):
    """Jina Embedding Provider."""

    PROVIDER_NAME = "JINA"

    def __init__(self, model: str = "jina-embeddings-v4", dim: int = 2048):
        super().__init__(model)
        self.dim = dim
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
            transport=RateLimitTransport(max_per_second=JINA_EMBEDDING_RPS),
        )
        self.url = "https://api.jina.ai/v1/embeddings"
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(JinaEmbeddingResponse)

    async def query(self, req: JinaEmbeddingRequest) -> JinaEmbeddingResponse:
        """Query the Jina embedding model with a request."""
        response = await self.client.post(self.url, content=self.encoder.encode(req))
        if response.is_error:
            raise HTTPCallError(
                "Failed to query Jina embedding", response.status_code, response.text
            )
        return self.decoder.decode(response.content)
