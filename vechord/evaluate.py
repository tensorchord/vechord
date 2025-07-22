from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

import msgspec
import pytrec_eval

from vechord.errors import DecodeStructuredOutputError
from vechord.model import GeminiGenerateRequest, RetrievedChunk, UMBRELAScore
from vechord.provider import GeminiGenerateProvider


class BaseEvaluator(ABC):
    def evaluate(
        self,
        chunk_ids: list[int],
        retrieves: list[list[RetrievedChunk]],
        measures: Sequence[str] = ("map", "ndcg", "recall"),
    ):
        """Evaluate the retrieval results for multiple queries."""
        num = len(chunk_ids)
        qids = list(range(num))
        query_relevance = {
            str(qid): {str(chunk_id): 1}
            for qid, chunk_id in zip(qids, chunk_ids, strict=False)
        }
        evaluator = pytrec_eval.RelevanceEvaluator(
            query_relevance=query_relevance, measures=measures
        )
        res = {
            str(qid): {str(r.uid): 1 / (r.score + 1e-6) for r in retrieve}
            for qid, retrieve in zip(qids, retrieves, strict=False)
        }
        evaluation = evaluator.evaluate(res)
        avg = defaultdict(float)
        for qres in evaluation.values():
            for k, v in qres.items():
                avg[k] = v
        for k in avg:
            avg[k] /= num
        return avg

    @staticmethod
    def evaluate_one(
        truth_id: str,
        resp_ids: list[str],
        measures: Sequence[str] = ("map", "ndcg", "recall"),
    ):
        """Evaluate the retrieval results for a single query."""
        query_relevance = {"0": {str(truth_id): 1}}
        evaluator = pytrec_eval.RelevanceEvaluator(
            query_relevance=query_relevance, measures=measures
        )
        res = {"0": {str(r): 1 / (i + 1) for i, r in enumerate(resp_ids)}}
        evaluation = evaluator.evaluate(res)
        return evaluation["0"]

    @staticmethod
    def calculate_avg_precision(is_relevant: list[bool], total: int) -> float:
        if total == 0:
            return 0.0

        precision = []
        relevant = 0
        for i, rel in enumerate(is_relevant, start=1):
            if rel:
                relevant += 1
                precision.append(relevant / i)
        return sum(precision) / len(precision)

    @staticmethod
    def calculate_mrr(is_relevant: list[bool]) -> float:
        for i, rel in enumerate(is_relevant, start=1):
            if rel:
                return 1.0 / i
        return 0.0

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class GeminiEvaluator(BaseEvaluator, GeminiGenerateProvider):
    """Evaluator using Gemini model to generate search queries."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        self.prompt = """
Given the following chunk of text and the overall document it belongs to, generate 
the most relevant and informative search query that accurately reflects the specific 
information contained within the chunk. Focus on creating a query that someone would 
use to find this exact information within the document. Prioritize using keywords 
and phrases directly from the chunk text. Consider the overall document context to 
refine the query and avoid ambiguity.
"""

    def name(self) -> str:
        return f"gemini_eval_{self.model}"

    async def produce_query(self, doc: str, chunk: str) -> str:
        contents = "\n".join(
            [
                self.prompt,
                f"<chunk> {chunk} </chunk>",
                f"<document> {doc} </document>",
            ]
        )
        resp = await self.query(GeminiGenerateRequest.from_prompt(contents))
        return resp.get_text().strip()


class GeminiUMBRELAEvaluator(BaseEvaluator, GeminiGenerateProvider):
    """Gemini evaluator with the Bing RELevance Assessor (UMBRELA) metric.

    - paper: https://arxiv.org/pdf/2406.06519
    """

    def __init__(self, model: str = "gemini-2.5-flash", relevant_threshold: int = 2):
        super().__init__(model)
        self.relevant_threshold = relevant_threshold
        self.k_values = (3, 5, 10)
        self.score_schema = msgspec.json.schema(UMBRELAScore)
        self.prompt = """
Given a query and a passage, you must provide a score on an
integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but
does not answer it,
2 = represents that the passage has some answer for the query,
but the answer may be a bit unclear, or hidden amongst extraneous
information and
3 = represents that the passage is dedicated to the query and
contains the exact answer.
Important Instruction: Assign category 1 if the passage is
somewhat related to the topic but not completely, category 2 if
passage presents something very important related to the entire
topic but also has some extra information and category 3 if the
passage only and entirely refers to the topic. If none of the
above satisfies give it category 0.
Query: {query}
Passage: {passage}
Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each,
and decide on a final score (O). Final score must be an integer.
Do not provide any code in result. Provide each score in the
format of: a single integer without any reasoning.
"""

    def name(self) -> str:
        return f"gemini_umbrela_{self.model}"

    async def estimate(self, query: str, passage: str) -> int:
        content = self.prompt.format(query=query, passage=passage)
        resp = await self.query(
            GeminiGenerateRequest.from_prompt_structure_response(
                content, self.score_schema
            )
        )
        try:
            score = msgspec.json.decode(resp.get_text(), type=UMBRELAScore).score
        except (msgspec.DecodeError, KeyError) as err:
            raise DecodeStructuredOutputError(
                "Failed to decode UMBRELA score from Gemini response",
            ) from err
        return score

    async def evaluate_with_estimation(
        self, query: str, passages: list[str]
    ) -> dict[str, float]:
        """Calculate the Precision@K and Mean Reciprocal Rank (MRR)."""
        scores = [await self.estimate(query, p) for p in passages]
        is_relevant = [score >= self.relevant_threshold for score in scores]
        metric = defaultdict(float)

        for k in self.k_values:
            if k > len(scores) or k <= 0:
                continue
            rel_at_k = sum(is_relevant[:k])
            metric[f"precision@{k}"] = rel_at_k / k
            metric[f"AP@{k}"] = self.calculate_avg_precision(is_relevant[:k], rel_at_k)

        metric["MRR"] = self.calculate_mrr(is_relevant)
        return metric
