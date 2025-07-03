from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from os import environ
from typing import TYPE_CHECKING, Annotated, Any, Optional

import msgspec

from vechord.chunk import GeminiChunker, RegexChunker
from vechord.client import VechordClient, limit_to_transaction_buffer
from vechord.embedding import BaseEmbedding, GeminiDenseEmbedding, OpenAIDenseEmbedding
from vechord.extract import GeminiExtractor
from vechord.model import ResourceRequest, RunAck, RunRequest
from vechord.rerank import CohereReranker
from vechord.spec import (
    DefaultDocument,
    Keyword,
    KeywordIndex,
    Vector,
    VectorIndex,
    _DefaultChunk,
)

if TYPE_CHECKING:
    from vechord.registry import VechordRegistry


class IndexOption:
    def __init__(self, vector=None, keyword=None):
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
    def __init__(self, vector=None, keyword=None):
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
    "ocr": {"gemini": GeminiExtractor},
    "rerank": {"cohere": CohereReranker},
    "index": {"vectorchord": IndexOption},
    "search": {"vectorchord": SearchOption},
}


@contextmanager
def set_api_key_in_env(keys: Iterable[str], values: Iterable[Optional[str]]):
    old_values = {key: environ.get(key) for key in keys if key}
    for key, value in zip(keys, values, strict=False):
        if not key:
            continue
        if value is None:
            environ.pop(key)
        environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                environ.pop(key, None)
            else:
                environ[key] = old_value


async def run_dynamic_pipeline(request: RunRequest, vr: "VechordRegistry"):  # noqa: PLR0912
    calls = build_pipeline(request.steps)
    emb: Optional[BaseEmbedding] = calls.get("embedding")
    if not emb:
        raise ValueError("No embedding provider specified in the request")
    dim = emb.get_dim()
    index: IndexOption = calls.get("index")
    search: SearchOption = calls.get("search")
    vr.reset_namespace(request.name)

    if index is None and search is None:
        raise ValueError("No index or search option specified in the request")

    # inject pipeline
    if index:
        vec_index = index.vector
        if vec_index is None:
            raise ValueError("No vector index specified in the request")
        use_keyword_index = index.keyword is not None

        class Chunk(_DefaultChunk):
            vec: Annotated[Vector[dim], vec_index]

        await vr.init_table_index([DefaultDocument, Chunk])

        # run the pipeline
        if ocr := calls.get("ocr"):
            text = await ocr.extract_pdf(request.data)
        else:
            text = request.data.decode("utf-8")
        if chunker := calls.get("chunk"):
            chunks = await chunker.segment(text)
        else:
            chunks = [text]
        vecs = []
        for chunk in chunks:
            vecs.append(await emb.vectorize_chunk(chunk))

        async with vr.client.transaction():
            doc = DefaultDocument(text=text)
            await vr.insert(doc)
            for i, vec in enumerate(vecs):
                await vr.insert(
                    Chunk(
                        vec=vec,
                        doc_id=doc.uid,
                        text=chunks[i],
                        keyword=Keyword(chunks[i]) if use_keyword_index else None,
                    )
                )
            return RunAck(name=request.name, msg="succeed", uid=doc.uid)
    elif search:
        query = request.data.decode("utf-8")

        class Chunk(_DefaultChunk):
            pass

        retrieved: list[Chunk] = []
        if vec_opt := search.vector:
            vec = await emb.vectorize_query(query)
            retrieved.extend(
                await vr.search_by_vector(Chunk, vec, vec_opt.topk, probe=vec_opt.probe)
            )
        if keyword_opt := search.keyword:
            retrieved.extend(await vr.search_by_keyword(Chunk, query, keyword_opt.topk))
        if rerank := calls.get("rerank"):
            indices = await rerank.rerank(query, [chunk.text for chunk in retrieved])
            retrieved = [retrieved[i] for i in indices]
        return retrieved


def build_pipeline(
    steps: list[ResourceRequest],
) -> dict[str, Callable]:
    calls = {}
    for step in steps:
        if step.kind not in PROVIDER_MAP:
            raise ValueError(f"Unsupported provider kind: {step.kind}")

        provider = PROVIDER_MAP[step.kind].get(step.provider)
        if not provider:
            raise ValueError(
                f"Unsupported provider: {step.provider} for kind: {step.kind}"
            )

        api_name, api_key = "", None
        args = step.args
        if "api_key" in args:
            api_name = f"{step.provider.upper()}_API_KEY"
            api_key = args.pop("api_key", None)
        with set_api_key_in_env((api_name,), (api_key,)):
            calls[step.kind] = provider(**args)
    return calls


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
