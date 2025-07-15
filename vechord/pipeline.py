from collections.abc import Callable, Iterable
from contextlib import contextmanager
from os import environ
from typing import TYPE_CHECKING, Annotated, Any, Optional

import msgspec

from vechord.chunk import GeminiChunker, RegexChunker
from vechord.client import VechordClient, limit_to_transaction_buffer_conn
from vechord.embedding import (
    BaseEmbedding,
    GeminiDenseEmbedding,
    JinaDenseEmbedding,
    OpenAIDenseEmbedding,
    VoyageDenseEmbedding,
    VoyageMultiModalEmbedding,
)
from vechord.errors import RequestError
from vechord.extract import GeminiExtractor
from vechord.model import InputType, ResourceRequest, RunAck, RunRequest
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


class GraphIndex(msgspec.Struct):
    enable_relation: bool = True


class IndexOption:
    def __init__(self, vector=None, keyword=None, graph=None):
        self.vector = (
            msgspec.convert(vector, VectorIndex) if vector is not None else None
        )
        self.keyword = (
            msgspec.convert(keyword, KeywordIndex) if keyword is not None else None
        )
        self.graph = msgspec.convert(graph, GraphIndex) if graph is not None else None


class VectorSearchOption(msgspec.Struct, kw_only=True):
    topk: int
    probe: Optional[int] = None


class KeywordSearchOption(msgspec.Struct, kw_only=True):
    topk: int


class GraphSearchOption(msgspec.Struct, kw_only=True):
    pass


class SearchOption:
    def __init__(self, vector=None, keyword=None):
        self.vector = msgspec.convert(vector, VectorSearchOption) if vector else None
        self.keyword = (
            msgspec.convert(keyword, KeywordSearchOption) if keyword else None
        )


PROVIDER_MAP: dict[str, dict[str, Any]] = {
    "chunk": {
        "regex": RegexChunker,
        "gemini": GeminiChunker,
    },
    "text-emb": {
        "gemini": GeminiDenseEmbedding,
        "jina": JinaDenseEmbedding,
        "openai": OpenAIDenseEmbedding,
        "voyage": VoyageDenseEmbedding,
    },
    "multimodal-emb": {"voyage": VoyageMultiModalEmbedding},
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


def build_pipeline(
    steps: list[ResourceRequest],
) -> dict[str, Callable]:
    calls = {}
    for step in steps:
        if step.kind not in PROVIDER_MAP:
            raise RequestError(f"Unsupported provider kind: {step.kind}")

        provider = PROVIDER_MAP[step.kind].get(step.provider)
        if not provider:
            raise RequestError(
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


async def run_index_pipeline(
    request: RunRequest, calls: dict[str, Callable], vr: "VechordRegistry"
):
    emb: Optional[BaseEmbedding] = calls.get("text-emb")
    multimodal: Optional[BaseEmbedding] = calls.get("multimodal-emb")
    dim = emb.get_dim() if emb else multimodal.get_dim()
    index: IndexOption = calls.get("index")
    vec_index = index.vector
    if vec_index is None:
        raise RequestError("No vector index specified in the request")
    enable_keyword_index = index.keyword is not None
    # enable_graph_index = index.graph is not None

    class Chunk(_DefaultChunk):
        vec: Annotated[Vector[dim], vec_index]

    await vr.init_table_index([DefaultDocument, Chunk])

    # run the pipeline
    if multimodal:
        vecs = [await multimodal.vectorize_multimodal_chunk(request.data)]
        text = ""
        chunks = [""]
    else:
        if request.input_type is InputType.TEXT:
            text = request.data.decode("utf-8")
        elif ocr := calls.get("ocr"):
            if request.input_type is InputType.PDF:
                text = await ocr.extract_pdf(request.data)
            elif request.input_type is InputType.IMAGE:
                text = await ocr.extract_image(request.data)
        else:
            raise RequestError(f"No OCR provider for input type: {request.input_type}")

        if chunker := calls.get("chunk"):
            chunks = await chunker.segment(text)
        else:
            chunks = [text]
        vecs = []
        for chunk in chunks:
            vecs.append(await emb.vectorize_chunk(chunk))

    async with (
        vr.client.get_connection() as conn,
        limit_to_transaction_buffer_conn(conn),
    ):
        doc = DefaultDocument(text=text)
        await vr.insert(doc)
        for i, vec in enumerate(vecs):
            await vr.insert(
                Chunk(
                    vec=vec,
                    doc_id=doc.uid,
                    text=chunks[i],
                    keyword=Keyword(chunks[i]) if enable_keyword_index else None,
                )
            )
        return RunAck(name=request.name, msg="succeed", uid=doc.uid)


async def run_dynamic_pipeline(request: RunRequest, vr: "VechordRegistry"):
    calls = build_pipeline(request.steps)
    emb: Optional[BaseEmbedding] = calls.get("text-emb")
    multimodal: Optional[BaseEmbedding] = calls.get("multimodal-emb")
    if not (emb or multimodal):
        raise RequestError("No embedding provider specified in the request")
    index: IndexOption = calls.get("index")
    search: SearchOption = calls.get("search")
    vr.reset_namespace(request.name)

    if index is None and search is None:
        raise RequestError("No `index` or `search` option specified in the request")

    # inject pipeline
    if index:
        return await run_index_pipeline(request, calls, vr)
    elif search:
        query = request.data.decode("utf-8")

        class Chunk(_DefaultChunk):
            pass

        retrieved: list[Chunk] = []
        if vec_opt := search.vector:
            emb_cls = emb if emb else multimodal
            vec = await emb_cls.vectorize_query(query)
            retrieved.extend(
                await vr.search_by_vector(Chunk, vec, vec_opt.topk, probe=vec_opt.probe)
            )
        if keyword_opt := search.keyword:
            retrieved.extend(await vr.search_by_keyword(Chunk, query, keyword_opt.topk))
        if rerank := calls.get("rerank"):
            indices = await rerank.rerank(query, [chunk.text for chunk in retrieved])
            retrieved = [retrieved[i] for i in indices]
        return retrieved


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
        async with (
            self.client.get_connection() as conn,
            limit_to_transaction_buffer_conn(conn),
        ):
            # only the 1st one can accept input (could be empty)
            await self.steps[0](*args, **kwargs)
            for func in self.steps[1:-1]:
                await func()
            return await self.steps[-1]()
