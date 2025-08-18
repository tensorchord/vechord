import inspect

import pytest

from vechord.client import set_namespace
from vechord.model import ResourceRequest, RunIngestAck, RunRequest
from vechord.pipeline import DynamicPipeline
from vechord.registry import VechordRegistry
from vechord.spec import DefaultDocument

pytestmark = pytest.mark.anyio


@pytest.fixture(name="run_pipeline_cleanup")
async def fixture_run_pipeline_cleanup(request, registry: VechordRegistry):
    namespace = request.node.obj.__name__
    yield
    # cleanup
    async with set_namespace(namespace):
        for table_name in ("defaultdocument", "chunk"):
            await registry.client.drop(table_name)


async def test_run_pipeline(registry: VechordRegistry, run_pipeline_cleanup):
    steps = [
        ResourceRequest(kind="text-emb", provider="spacy", args={}),
        ResourceRequest(
            kind="chunk", provider="regex", args={"size": 128, "overlap": 0}
        ),
        ResourceRequest(kind="index", provider="vectorchord", args={"vector": {}}),
    ]
    namespace = inspect.currentframe().f_code.co_name
    pipe = DynamicPipeline.from_steps(steps=steps)
    ack: RunIngestAck = await pipe.run(
        RunRequest(name=namespace, data="what to insert".encode(), steps=steps),
        vr=registry,
    )
    assert ack.name == namespace
    assert ack.uid

    docs = await registry.select_by(DefaultDocument.partial_init())
    assert len(docs) == 1
