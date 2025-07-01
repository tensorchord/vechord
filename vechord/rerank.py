from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AsyncExitStack
from os import environ
from typing import TypeVar

from vechord.spec import Table

T = TypeVar("T", bound=Table)


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, chunks: list[str]) -> list[int]:
        """Return the indices of the reranked chunks."""
        raise NotImplementedError


class CohereReranker(BaseReranker):
    """Rerank chunks using Cohere API (requires env `COHERE_API_KEY`)."""

    def __init__(self, model: str = "rerank-v3.5"):
        self.api_key = environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not found in environment variables.")

        import cohere

        self.client = cohere.AsyncClientV2(api_key=self.api_key)
        self.model = model

    async def __aenter__(self):
        self._async_exit_stack = AsyncExitStack()
        await self._async_exit_stack.enter_async_context(self.client)
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self._async_exit_stack.aclose()

    async def rerank(self, query: str, chunks: list[str]) -> list[int]:
        resp = await self.client.rerank(
            model=self.model,
            query=query,
            documents=chunks,
        )
        return [item.index for item in resp.results]


class ReciprocalRankFusion:
    """Fuse chunks using reciprocal rank."""

    def __init__(self, k: int = 60):
        self.k = k

    def get_score(self, rank: int) -> float:
        return 1 / (self.k + rank)

    def fuse(self, retrieved_chunks: list[list[T]]) -> list[T]:
        chunk_score: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Table] = {}
        for chunks in retrieved_chunks:
            for i, chunk in enumerate(chunks):
                chunk_score[getattr(chunk, chunk.primary_key())] += self.get_score(i)
                chunk_map[getattr(chunk, chunk.primary_key())] = chunk

        sorted_uid = sorted(chunk_score, key=lambda x: chunk_score[x], reverse=True)
        return [chunk_map[uid] for uid in sorted_uid]
