import uuid
from unittest.mock import AsyncMock

import pytest

from vechord.client import set_namespace
from vechord.groundtruth import GroundTruth
from vechord.registry import VechordRegistry
from vechord.spec import _DefaultChunk

pytestmark = pytest.mark.anyio


@pytest.fixture(name="ground_truth_cleanup")
async def fixture_ground_truth_cleanup(request, registry: VechordRegistry):
    namespace = request.node.obj.__name__
    yield
    # cleanup
    async with set_namespace(namespace):
        await registry.client.drop("test_query")


async def test_ground_truth(registry: VechordRegistry, ground_truth_cleanup):
    queries = [
        "What is the largest mammal?",
        "What is the longest river in the world?",
        "What is the smallest bird?",
    ]

    async def mock_retrieve(query: str):
        return [
            _DefaultChunk(
                uid=uuid.uuid5(uuid.NAMESPACE_DNS, query),
                doc_id=None,
                text=query,
                vec=None,
                keyword=None,
            )
        ]

    async def mock_estimate(query: str, passage: str, chunk_type=None):
        return 1.0 + 2.0 if query == passage else 0.0

    retrieve = AsyncMock()
    retrieve.side_effect = mock_retrieve
    evaluator = AsyncMock()
    evaluator.estimate = mock_estimate
    evaluator.relevant_threshold = 2.0

    ground_truth = GroundTruth(name="test", vr=registry)
    await ground_truth.generate(queries, retrieve, evaluator)

    assert retrieve.call_count == len(queries)

    # evaluate
    metric = await ground_truth.evaluate(retrieve=retrieve)
    assert metric.ndcg == 1.0, metric
