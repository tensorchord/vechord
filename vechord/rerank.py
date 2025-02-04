from abc import ABC, abstractmethod
from collections import defaultdict

from vechord.model import RetrivedChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, chunks: list[str]) -> list[str]:
        raise NotImplementedError


class CohereReranker(BaseReranker):
    def __init__(self):
        super().__init__()

    def rerank(self, chunks: list[str]) -> list[str]:
        return super().rerank(chunks)


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def get_score(self, rank: int) -> float:
        return 1 / (self.k + rank)

    def fuse(self, retrived_chunks: list[list[RetrivedChunk]]) -> list[RetrivedChunk]:
        chunk_score: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrivedChunk] = {}
        for retrives in retrived_chunks:
            for i, retrive in enumerate(retrives):
                chunk_score[retrive.uid] += self.get_score(i)
                chunk_map[retrive.uid] = retrive

        sorted_uid = sorted(chunk_score, key=lambda x: chunk_score[x], reverse=True)
        return [chunk_map[uid] for uid in sorted_uid]
