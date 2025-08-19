from collections.abc import Awaitable, Callable
from typing import Iterable

import msgspec

from vechord.client import limit_to_transaction_buffer_conn
from vechord.evaluate import BaseEvaluator, GeminiUMBRELAEvaluator
from vechord.model import InputType
from vechord.registry import VechordRegistry
from vechord.spec import PrimaryKeyUUID, Table


class Query(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    truth: list[str] = msgspec.field(default_factory=list)


class Metric(msgspec.Struct):
    ndcg: float
    map: float
    recall: float


RetrieveFunc = Callable[[str], Awaitable[Iterable[Table]]]


class GroundTruth:
    """Generate the ground truth for evaluation."""

    def __init__(self, name: str, vr: VechordRegistry):
        self.name = name
        self.vr = vr
        self.query_cls = msgspec.defstruct(f"{name}_query", (), bases=(Query,))

    async def generate(
        self,
        queries: list[str],
        retrieve: RetrieveFunc,
        evaluator: GeminiUMBRELAEvaluator,
        chunk_type: InputType = InputType.TEXT,
    ):
        """Generate the ground truth from the retrieved chunks.

        **Note**: This assumes that the ground truth can be obtained from the retrieval results.

        **Warning**: If the chunk table has updated, the ground truth might be outdated.

        Args:
            queries: list of text queries.
            retrieve: a function `async fn(query: str) -> [Chunk]`
            evaluator: the UMBRELA evaluator to estimate the relevance score
            chunk_type: used by evaluator

        Examples:
            .. code-block:: python

                async def retrieve_fn(query: str) -> list[Chunk]:
                    return await vr.search_by_vector(Chunk, emb.vectorize_query(query), topk=100)

                gt = GroundTruth("record-1", vr)
                gt.generate(
                    queries=["What's the longest river?", "What's the largest ocean?"],
                    retrieve=retrieve_fn,
                    evaluator=GeminiUMBRELAEvaluator(),
                )

        """
        await self.vr.init_table_index((self.query_cls,))
        async with (
            self.vr.client.get_connection() as conn,
            limit_to_transaction_buffer_conn(conn),
        ):
            query_truth = []
            for query in queries:
                top_chunks = await retrieve(query)
                scores: list[tuple[int, int]] = []
                for i, chunk in enumerate(top_chunks):
                    scores.append(
                        (
                            await evaluator.estimate(
                                query=query,
                                passage=chunk.text,
                                chunk_type=chunk_type,
                            ),
                            i,
                        )
                    )
                indices = [
                    x[1]
                    for x in sorted(
                        filter(lambda x: x[0] >= evaluator.relevant_threshold, scores)
                    )
                ]
                query_truth.append(
                    self.query_cls(
                        text=query,
                        truth=[str(top_chunks[i].uid) for i in indices],
                    )
                )

            await self.vr.copy_bulk(query_truth)

    async def evaluate(self, retrieve: RetrieveFunc, topk: int = 10):
        """Evaluate with a retrieve function.

        Args:
            retrieve: a function `async fn(query: str) -> [Chunk]`
            topk: recall@k value
        """
        queries: list[Query] = await self.vr.select_by(self.query_cls.partial_init())
        metric = []
        recall_key = f"recall_{topk}"
        for query in queries:
            retrieved_chunks = await retrieve(query.text)
            m = BaseEvaluator.evaluate_one(
                query.truth, [str(chunk.uid) for chunk in retrieved_chunks]
            )
            metric.append(
                Metric(ndcg=m.get("ndcg"), map=m.get("map"), recall=m.get(recall_key))
            )

        return Metric(
            ndcg=sum(m.ndcg for m in metric) / len(metric),
            map=sum(m.map for m in metric) / len(metric),
            recall=sum(m.recall for m in metric) / len(metric),
        )
