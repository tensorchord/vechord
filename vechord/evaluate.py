import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

import pytrec_eval

from vechord.model import RetrievedChunk


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
        truth_id: int,
        resp_ids: list[int],
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

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def produce_query(self, doc: str, chunk: str) -> str:
        raise NotImplementedError


class GeminiEvaluator(BaseEvaluator):
    """Evaluator using Gemini model to generate search queries."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("env GEMINI_API_KEY not set")

        import google.generativeai as genai

        self.model = genai.GenerativeModel(model)
        self.prompt = """
Given the following chunk of text and the overall document it belongs to, generate 
the most relevant and informative search query that accurately reflects the specific 
information contained within the chunk. Focus on creating a query that someone would 
use to find this exact information within the document. Prioritize using keywords 
and phrases directly from the chunk text. Consider the overall document context to 
refine the query and avoid ambiguity.
"""

    def name(self) -> str:
        return f"gemini_eval_{self.model.model_name}"

    def produce_query(self, doc: str, chunk: str) -> str:
        resp = self.model.generate_content(
            contents="\n".join(
                [
                    self.prompt,
                    f"<chunk> {chunk} </chunk>",
                    f"<document> {doc} </document>",
                ]
            )
        )
        return resp.text
