from abc import ABC, abstractmethod
from collections import defaultdict

from vechord.spec import Table


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[str]) -> list[str]:
        """Return the indices of the reranked chunks."""
        raise NotImplementedError


class CohereReranker(BaseReranker):
    """Rerank chunks using Cohere API (requires env `COHERE_API_KEY`)."""

    def __init__(self, model: str = "rerank-v3.5"):
        import cohere

        self.client = cohere.ClientV2()
        self.model = model

    def rerank(self, query: str, chunks: list[str]) -> list[int]:
        resp = self.client.rerank(
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

    def fuse(self, retrieved_chunks: list[list[Table]]) -> list[Table]:
        chunk_score: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Table] = {}
        for chunks in retrieved_chunks:
            for i, chunk in enumerate(chunks):
                chunk_score[chunk.uid] += self.get_score(i)
                chunk_map[chunk.uid] = chunk

        sorted_uid = sorted(chunk_score, key=lambda x: chunk_score[x], reverse=True)
        return [chunk_map[uid] for uid in sorted_uid]
