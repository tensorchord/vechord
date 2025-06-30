from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import msgspec

from vechord.chunk import GeminiChunker, RegexChunker
from vechord.client import VechordClient, limit_to_transaction_buffer
from vechord.embedding import GeminiDenseEmbedding, OpenAIDenseEmbedding
from vechord.extract import GeminiExtractor
from vechord.model import ResourceRequest
from vechord.rerank import CohereReranker
from vechord.spec import KeywordIndex, VectorIndex


class IndexOption:
    def __init__(self, vector, keyword):
        self.vector = msgspec.convert(vector, VectorIndex) if vector else None
        self.keyword = msgspec.convert(keyword, KeywordIndex) if keyword else None


@dataclass
class VectorSearchOption:
    topk: int
    probe: Optional[int] = None


@dataclass
class KeywordSearchOption:
    topk: int


class SearchOption:
    def __init__(self, vector, keyword):
        self.vector = msgspec.convert(vector, VectorSearchOption) if vector else None
        self.keyword = (
            msgspec.convert(keyword, KeywordSearchOption) if keyword else None
        )


PROVIDER_MAP = {
    "chunk": {
        "regex": RegexChunker,
        "gemini": GeminiChunker,
    },
    "embedding": {
        "gemini": GeminiDenseEmbedding,
        "openai": OpenAIDenseEmbedding,
    },
    "extract": {"gemini": GeminiExtractor},
    "rerank": {"cohere": CohereReranker},
    "index": {"vectorchord": IndexOption},
    "search": {"vectorchord": SearchOption},
}


def build_pipeline(steps: list[ResourceRequest]) -> Callable:
    calls = []
    for step in steps:
        if step.kind not in PROVIDER_MAP:
            raise ValueError(f"Unsupported provider kind: {step.kind}")

        provider = PROVIDER_MAP[step.kind].get(step.provider)
        if not provider:
            raise ValueError(
                f"Unsupported provider: {step.provider} for kind: {step.kind}"
            )

        calls.append(provider)


class VechordPipeline:
    """Set up the pipeline to run multiple functions in a transaction.

    Args:
        client: :class:`VectorChordClient` to be used for the transaction.
        steps: a list of functions to be run in the pipeline. The first function
            will be used to accept the input, and the last function will be used
            to return the output. The rest of the functions will be used to
            process the data in between. The functions will be run in the order
            they are defined in the list.
    """

    def __init__(self, client: VechordClient, steps: list[Callable]):
        self.client = client
        self.steps = steps

    async def run(self, *args, **kwargs) -> Any:
        """Execute the pipeline in a transactional manner.

        All the `args` and `kwargs` will be passed to the first function in the
        pipeline. The pipeline will run in *one* transaction, and all the `inject`
        can only see the data inserted in this transaction (to guarantee only the
        new inserted data will be processed in this pipeline).

        This will also return the final result of the last function in the pipeline.
        """
        async with self.client.transaction():
            with limit_to_transaction_buffer():
                # only the 1st one can accept input (could be empty)
                await self.steps[0](*args, **kwargs)
                for func in self.steps[1:-1]:
                    await func()
                return await self.steps[-1]()
