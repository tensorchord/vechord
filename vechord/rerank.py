from abc import ABC, abstractmethod
from collections import defaultdict

from vechord.model import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, chunks: list[str]) -> list[str]:
        raise NotImplementedError


class CohereReranker(BaseReranker):
    def __init__(self):
        super().__init__()

    def rerank(self, chunks: list[str]) -> list[str]:
        raise NotImplementedError


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def get_score(self, rank: int) -> float:
        return 1 / (self.k + rank)

    def fuse(
        self, retrieved_chunks: list[list[RetrievedChunk]]
    ) -> list[RetrievedChunk]:
        chunk_score: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}
        for chunks in retrieved_chunks:
            for i, chunk in enumerate(chunks):
                chunk_score[chunk.uid] += self.get_score(i)
                chunk_map[chunk.uid] = chunk

        sorted_uid = sorted(chunk_score, key=lambda x: chunk_score[x], reverse=True)
        return [chunk_map[uid] for uid in sorted_uid]
