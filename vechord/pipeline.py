import itertools
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from os import environ
from typing import TYPE_CHECKING, Annotated, Any, Optional
from uuid import UUID

import msgspec

from vechord.chunk import BaseChunker, GeminiChunker, RegexChunker
from vechord.client import VechordClient, limit_to_transaction_buffer_conn
from vechord.embedding import (
    BaseEmbedding,
    GeminiDenseEmbedding,
    JinaDenseEmbedding,
    OpenAIDenseEmbedding,
    VoyageDenseEmbedding,
    VoyageMultiModalEmbedding,
)
from vechord.entity import GeminiEntityRecognizer
from vechord.errors import RequestError
from vechord.extract import GeminiExtractor
from vechord.model import (
    InputType,
    ResourceRequest,
    RunAck,
    RunRequest,
)
from vechord.rerank import CohereReranker
from vechord.spec import (
    DefaultDocument,
    Keyword,
    KeywordIndex,
    PrimaryKeyUUID,
    Table,
    Vector,
    VectorIndex,
    _DefaultChunk,
)
from vechord.typing import Self

if TYPE_CHECKING:
    from vechord.registry import VechordRegistry


class GraphIndex(msgspec.Struct):
    """Graph index for entities and relations extracted from the text/image."""


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
    topk: int


class SearchOption:
    def __init__(self, vector=None, keyword=None, graph=None):
        self.vector = msgspec.convert(vector, VectorSearchOption) if vector else None
        self.keyword = (
            msgspec.convert(keyword, KeywordSearchOption) if keyword else None
        )
        self.graph = msgspec.convert(graph, GraphSearchOption) if graph else None


class _Entity(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    chunk_uuids: list[UUID]
    text: str
    label: str
    description: str = ""
    vec: Vector[1]


class _Relation(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    source: UUID
    target: UUID
    description: str
    vec: Vector[1]


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
    "graph": {"gemini": GeminiEntityRecognizer},
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


def find_uid_by_text(text: str, ents: list[_Entity]) -> Optional[UUID]:
    for ent in ents:
        if ent.text == text:
            return ent.uid
    return None


class DynamicPipeline(msgspec.Struct, kw_only=True):
    chunk: Optional[BaseChunker] = None
    text_emb: Optional[BaseEmbedding] = None
    multimodal_emb: Optional[BaseEmbedding] = None
    ocr: Optional[GeminiExtractor] = None
    rerank: Optional[CohereReranker] = None
    index: Optional[IndexOption] = None
    search: Optional[SearchOption] = None
    graph: Optional[GeminiEntityRecognizer] = None

    def __post_init__(self):
        if not (self.text_emb or self.multimodal_emb):
            raise RequestError("No embedding provider specified in the request")
        if self.index is None and self.search is None:
            raise RequestError("No `index` or `search` option specified in the request")
        if self.index and self.index.graph and not self.graph:
            raise RequestError("Graph index requires a graph provider")

    @classmethod
    def from_steps(cls, steps: list[ResourceRequest]) -> Self:
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
                calls[(step.kind).replace("-", "_")] = provider(**args)
        return msgspec.convert(calls, DynamicPipeline)

    def run(self, request: RunRequest, vr: "VechordRegistry"):
        """Run the dynamic pipeline with the given request."""
        if self.index:
            return self.run_index(request, vr)
        elif self.search:
            return self.run_search(request, vr)
        else:
            raise RequestError("No valid pipeline configuration found")

    async def run_index(self, request: RunRequest, vr: "VechordRegistry") -> RunAck:  # noqa: PLR0912
        dim = (
            self.text_emb.get_dim() if self.text_emb else self.multimodal_emb.get_dim()
        )
        vec_index = self.index.vector
        if vec_index is None:
            raise RequestError("No vector index specified in the request")
        enable_keyword_index = self.index.keyword is not None
        enable_graph_index = self.index.graph is not None
        vec_index_type = Annotated[Vector[dim], vec_index]

        Chunk = msgspec.defstruct(
            "Chunk", (("vec", vec_index_type),), bases=(_DefaultChunk,)
        )
        # use the default vector index for Entity and Relation
        Entity = msgspec.defstruct("Entity", (("vec", Vector[dim]),), bases=(_Entity,))
        Relation = msgspec.defstruct(
            "Relation", (("vec", Vector[dim]),), bases=(_Relation,)
        )
        tables = [DefaultDocument, Chunk]
        if enable_graph_index:
            tables.extend([Entity, Relation])

        await vr.init_table_index(tables=tables)

        # run the pipeline
        sentences, chunks, ents, rels = [], [], [], []
        doc = DefaultDocument(text="")
        if self.multimodal_emb:
            chunks.append(
                Chunk(
                    doc_id=doc.uid,
                    text="",
                    keyword=None,
                    vec=await self.multimodal_emb.vectorize_multimodal_chunk(
                        request.data
                    ),
                )
            )
        else:
            if request.input_type is InputType.TEXT:
                doc.text = request.data.decode("utf-8")
            elif self.ocr:
                if request.input_type is InputType.PDF:
                    doc.text = await self.ocr.extract_pdf(request.data)
                elif request.input_type is InputType.IMAGE:
                    doc.text = await self.ocr.extract_image(request.data)
            elif self.graph:
                img_ents, img_rels = await self.graph.recognize_image(request.data)
                ents.extend(
                    [
                        Entity(
                            chunk_uuids=[doc.uid],
                            text=ent.text,
                            label=ent.label,
                            description=ent.description,
                            vec=None,  # will be set later
                        )
                        for ent in img_ents
                    ]
                )
                rels.extend(
                    [
                        Relation(
                            source=find_uid_by_text(rel.source.text, ents),
                            target=find_uid_by_text(rel.target.text, ents),
                            description=rel.description,
                            vec=None,  # will be set later
                        )
                        for rel in img_rels
                    ]
                )
            else:
                raise RequestError(
                    f"No OCR provider for input type: {request.input_type}"
                )

            if self.chunk:
                sentences.extend(await self.chunk.segment(doc.text))
            elif doc.text:
                sentences.append(doc.text)
            for sent in sentences:
                chunk = Chunk(
                    vec=await self.text_emb.vectorize_chunk(sent),
                    doc_id=doc.uid,
                    text=sent,
                    keyword=None if not enable_keyword_index else Keyword(sent),
                )
                if self.graph and request.input_type is InputType.TEXT:
                    chunk_ents, chunk_rels = await self.graph.recognize_with_relations(
                        sent
                    )
                    ents.extend(
                        Entity(
                            chunk_uuids=[chunk.uid],
                            text=ent.text,
                            label=ent.label,
                            description=ent.description,
                            vec=None,  # will be set later
                        )
                        for ent in chunk_ents
                    )
                    rels.extend(
                        Relation(
                            source=find_uid_by_text(rel.source.text, ents),
                            target=find_uid_by_text(rel.target.text, ents),
                            description=rel.description,
                            vec=None,  # will be set later
                        )
                        for rel in chunk_rels
                    )
                chunks.append(chunk)

        async with (
            vr.client.get_connection() as conn,
            limit_to_transaction_buffer_conn(conn),
        ):
            await vr.insert(doc)
            for chunk in enumerate(chunks):
                await vr.insert(chunk)
            if self.index.graph:
                await self.graph_insert(
                    ents=ents, rels=rels, ent_cls=Entity, rel_cls=Relation, vr=vr
                )
            return RunAck(name=request.name, msg="succeed", uid=doc.uid)

    async def graph_insert(
        self,
        ents: list[_Entity],
        rels: list[_Relation],
        ent_cls: type[Table],
        rel_cls: type[Table],
        vr: "VechordRegistry",
    ):
        """Insert entities and relations into the graph index."""
        ent_map: dict[str, _Entity] = {}
        for ent in ents:
            if ent.text not in ent_map:
                ent_map[ent.text] = ent
            else:
                ent_map[ent.text].chunk_uuids.extend(ent.chunk_uuids)
                ent_map[ent.text].description += f"\n{ent.description}"
        for ent in ent_map.values():
            exist_ent = await vr.select_by(
                ent_cls.partial_init(text=ent.text),
                fields=("uid", "description", "chunk_uuids"),
            )
            if exist_ent:
                exist = exist_ent[0]
                ent.chunk_uuids.extend(exist.chunk_uuids)
                ent.description += f"\n{exist.description}"
                await vr.remove_by(ent_cls.partial_init(uid=exist.uid))
            ent.vec = await self.text_emb.vectorize_chunk(
                f"{ent.text}\n{ent.description}"
            )
            await vr.insert(ent)

        relation_map: dict[str, _Relation] = {}
        for rel in rels:
            key = "|".join(sorted(map(str, [rel.source, rel.target])))
            if key not in relation_map:
                relation_map[key] = rel
            else:
                relation_map[key].description += f"\n{rel.description}"
        for rel in relation_map.values():
            exist_rel = await vr.select_by(
                rel_cls.partial_init(source=rel.source, target=rel.target),
                fields=("uid", "description"),
            )
            if exist_rel:
                exist = exist_rel[0]
                rel.description += f"\n{exist.description}"
                await vr.remove_by(rel_cls.partial_init(uid=exist.uid))
            await vr.insert(rel)

    async def run_search(self, request: RunRequest, vr: "VechordRegistry"):
        query = request.data.decode("utf-8")

        # for type hint and compatibility
        class Chunk(_DefaultChunk):
            pass

        class Entity(_Entity):
            pass

        class Relation(_Relation):
            pass

        retrieved: list[Chunk] = []
        if self.search.vector:
            emb_cls = self.text_emb if self.text_emb else self.multimodal_emb
            vec = await emb_cls.vectorize_query(query)
            retrieved.extend(
                await vr.search_by_vector(
                    Chunk, vec, self.search.vector.topk, probe=self.search.vector.probe
                )
            )
        if self.search.keyword:
            retrieved.extend(
                await vr.search_by_keyword(Chunk, query, self.search.keyword.topk)
            )
        if self.search.graph:
            retrieved.extend(
                await self.graph_search(query, Chunk, Entity, Relation, vr)
            )
        if self.rerank:
            indices = await self.rerank.rerank(
                query, [chunk.text for chunk in retrieved]
            )
            retrieved = [retrieved[i] for i in indices]
        return retrieved

    async def graph_search(
        self,
        query: str,
        chunk_cls: type[Table],
        ent_cls: type[Table],
        rel_cls: type[Table],
        vr: "VechordRegistry",
    ):
        ents, rels = await self.graph.recognize_with_relations(query)
        if rels:
            rel_text = " ".join(rel.description for rel in rels)
            similar_rels = await vr.search_by_vector(
                rel_cls,
                await self.text_emb.vectorize_query(rel_text),
                topk=self.search.graph.topk,
            )
            ent_uuids = deduplicate_uid(
                itertools.chain.from_iterable(
                    (rel.source, rel.target) for rel in similar_rels
                )
            )
            ents.extend(
                vr.select_by(
                    ent_cls.partial_init(uuid=ent_uuid), fields=("text", "description")
                )
                for ent_uuid in ent_uuids
            )
        if not ents:
            return []
        ent_text = " ".join(f"{ent.text} {ent.description}" for ent in ents)
        similar_ents = await vr.search_by_vector(
            ent_cls,
            await self.text_emb.vectorize_query(ent_text),
            topk=self.search.graph.topk,
        )
        chunk_uuids = deduplicate_uid(
            itertools.chain.from_iterable(ent.chunk_uuids for ent in similar_ents),
            limit=self.search.graph.topk,
        )
        return itertools.chain.from_iterable(
            [
                vr.select_by(
                    chunk_cls.partial_init(uid=chunk_uuid),
                    fields=("text", "doc_id", "uid"),
                )
                for chunk_uuid in chunk_uuids
            ]
        )


def deduplicate_uid(uuids: Iterable[UUID], limit: Optional[int] = None) -> list[UUID]:
    uuids = {uid: None for uid in uuids}
    return list(uuids.keys())[:limit]


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
