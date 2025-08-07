import base64
import itertools
from collections.abc import Iterable
from contextlib import contextmanager
from os import environ
from typing import Annotated, Any, Optional
from uuid import UUID

import msgspec

from vechord.chunk import BaseChunker, GeminiChunker, RegexChunker
from vechord.client import (
    set_namespace,
)
from vechord.embedding import (
    BaseMultiModalEmbedding,
    BaseTextEmbedding,
    GeminiDenseEmbedding,
    JinaDenseEmbedding,
    JinaMultiModalEmbedding,
    OpenAIDenseEmbedding,
    VoyageDenseEmbedding,
    VoyageMultiModalEmbedding,
)
from vechord.errors import RequestError
from vechord.evaluate import GeminiUMBRELAEvaluator
from vechord.extract import GeminiExtractor, LlamaParseExtractor
from vechord.graph import GeminiEntityRecognizer
from vechord.model import (
    GraphEntity,
    GraphRelation,
    InputType,
    ResourceRequest,
    RunIngestAck,
    RunRequest,
    RunSearchResponse,
)
from vechord.registry import VechordRegistry
from vechord.rerank import BaseReranker, CohereReranker, JinaReranker
from vechord.spec import (
    AnyOf,
    DefaultDocument,
    Keyword,
    KeywordIndex,
    PrimaryKeyUUID,
    RunChunk,
    Table,
    Vector,
    VectorIndex,
)
from vechord.typing import Self


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
    topk: Optional[int] = 10
    probe: Optional[int] = None


class KeywordSearchOption(msgspec.Struct, kw_only=True):
    topk: Optional[int] = 10


class GraphSearchOption(msgspec.Struct, kw_only=True):
    topk: Optional[int] = 10
    "final topk for the search result"
    similar_k: Optional[int] = 10
    "how many similar entities/relationships to retrieve during the graph search"


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
    vec: Vector[1]  # as a placeholder, will be replaced by the actual vector type


class _Relation(Table, kw_only=True):
    uid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    source: UUID
    target: UUID
    description: str
    vec: Vector[1]  # as a placeholder, will be replaced by the actual vector type


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
    "multimodal-emb": {
        "voyage": VoyageMultiModalEmbedding,
        "jina": JinaMultiModalEmbedding,
    },
    "ocr": {"gemini": GeminiExtractor, "llamaparse": LlamaParseExtractor},
    "rerank": {"cohere": CohereReranker, "jina": JinaReranker},
    "graph": {"gemini": GeminiEntityRecognizer},
    "index": {"vectorchord": IndexOption},
    "search": {"vectorchord": SearchOption},
    "evaluate": {"gemini": GeminiUMBRELAEvaluator},
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
    text_emb: Optional[BaseTextEmbedding] = None
    multimodal_emb: Optional[BaseMultiModalEmbedding] = None
    ocr: Optional[GeminiExtractor] = None
    rerank: Optional[BaseReranker] = None
    index: Optional[IndexOption] = None
    search: Optional[SearchOption] = None
    graph: Optional[GeminiEntityRecognizer] = None
    evaluate: Optional[GeminiUMBRELAEvaluator] = None

    def __post_init__(self):
        if not (self.text_emb or self.multimodal_emb):
            raise RequestError("No embedding provider specified in the request")
        if self.index is None and self.search is None:
            raise RequestError("No `index` or `search` option specified in the request")
        if self.index and self.index.graph and not self.graph:
            raise RequestError("Graph index requires a graph provider")
        if self.index and self.index.vector is None:
            raise RequestError("Vector index is required if `index` is specified")
        if self.search and not (self.text_emb or self.multimodal_emb):
            raise RequestError("Search requires at least one embedding provider")

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

    async def run(
        self, request: RunRequest, vr: VechordRegistry
    ) -> RunIngestAck | RunSearchResponse:
        """Run the dynamic pipeline with the given request."""
        async with set_namespace(request.name):
            resp = (
                await self.run_index(request, vr)
                if self.index
                else await self.run_search(request, vr)
            )
            return resp

    @staticmethod
    def _convert_from_extracted_graph(
        uid: UUID,
        ents: list[GraphEntity],
        rels: list[GraphRelation],
        ent_cls: type[Table],
        rel_cls: type[Table],
    ):
        converted_ents = [
            ent_cls(
                chunk_uuids=[uid],
                text=ent.text,
                label=ent.label,
                description=ent.description,
                vec=None,  # will be set later
            )
            for ent in ents
        ]
        converted_rels = [
            rel_cls(
                source=find_uid_by_text(rel.source.text, converted_ents),
                target=find_uid_by_text(rel.target.text, converted_ents),
                description=f"{rel.source.text} {rel.description} {rel.target.text}",
                vec=None,  # will be set later
            )
            for rel in rels
        ]
        return converted_ents, converted_rels

    async def run_index(  # noqa: PLR0912
        self, request: RunRequest, vr: VechordRegistry
    ) -> RunIngestAck:
        dim = (
            self.text_emb.get_dim() if self.text_emb else self.multimodal_emb.get_dim()
        )
        enable_keyword_index = self.index.keyword is not None
        enable_graph_index = self.index.graph is not None
        vec_index_type = Annotated[Vector[dim], self.index.vector]

        Chunk = msgspec.defstruct(
            "Chunk", (("vec", vec_index_type),), bases=(RunChunk,)
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
        # Create a fake chunk to ref the document by doc_id.
        # This is useful because graph entities and relations will ref the chunk uuids
        # and image/pdf doesn't have real chunks.
        fake_chunk = Chunk(
            doc_id=doc.uid,
            text="",
            text_type=request.input_type.value,
            keyword=None,
            vec=None,
        )
        if self.multimodal_emb and request.input_type is not InputType.TEXT:
            # reuse the fake chunk to ensure the chunk uid is unique
            fake_chunk.text = base64.b64encode(request.data).decode("utf-8")
            fake_chunk.vec = await self.multimodal_emb.vectorize_multimodal_chunk(
                image=request.data
            )
            chunks.append(fake_chunk)
        if request.input_type is InputType.TEXT:
            doc.text = request.data.decode("utf-8")
        elif self.ocr:
            if request.input_type is InputType.PDF:
                doc.text = await self.ocr.extract_pdf(request.data)
            elif request.input_type is InputType.IMAGE:
                doc.text = await self.ocr.extract_image(request.data)
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
            chunks.append(chunk)
            if self.graph and request.input_type is InputType.TEXT:
                chunk_ents, chunk_rels = await self.graph.recognize_with_relations(sent)
                conv_ents, conv_rels = self._convert_from_extracted_graph(
                    chunk.uid, chunk_ents, chunk_rels, Entity, Relation
                )
                ents.extend(conv_ents)
                rels.extend(conv_rels)

        if self.graph and request.input_type is not InputType.TEXT and not sentences:
            img_ents, img_rels = await self.graph.recognize_image(request.data)
            conv_ents, conv_rels = self._convert_from_extracted_graph(
                fake_chunk.uid, img_ents, img_rels, Entity, Relation
            )
            ents.extend(conv_ents)
            rels.extend(conv_rels)
            if not self.multimodal_emb:
                chunks.append(fake_chunk)

        await vr.insert(doc)
        for chunk in chunks:
            await vr.insert(chunk)
        if self.index.graph:
            await self.graph_insert(
                ents=ents, rels=rels, ent_cls=Entity, rel_cls=Relation, vr=vr
            )
        return RunIngestAck(name=request.name, msg="succeed", uid=doc.uid)

    async def graph_insert(
        self,
        ents: list[_Entity],
        rels: list[_Relation],
        ent_cls: type[Table],
        rel_cls: type[Table],
        vr: VechordRegistry,
    ):
        """Insert entities and relations into the graph index."""
        ent_map: dict[str, _Entity] = {}
        emb_func = (
            self.text_emb.vectorize_chunk
            if self.text_emb
            else self.multimodal_emb.vectorize_multimodal_chunk
        )
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
            ent.vec = await emb_func(f"{ent.text}\n{ent.description}")
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
            rel.vec = await emb_func(f"{rel.description}")
            await vr.insert(rel)

    async def run_search(
        self, request: RunRequest, vr: VechordRegistry
    ) -> RunSearchResponse:
        query = request.data.decode("utf-8")

        # for type hint and compatibility
        class Chunk(RunChunk):
            pass

        class Entity(_Entity):
            pass

        class Relation(_Relation):
            pass

        resp = RunSearchResponse()
        if self.search.vector:
            vec = (
                await self.text_emb.vectorize_query(query)
                if self.text_emb
                else await self.multimodal_emb.vectorize_multimodal_query(text=query)
            )
            resp.extend(
                await vr.search_by_vector(
                    Chunk, vec, self.search.vector.topk, probe=self.search.vector.probe
                )
            )
        if self.search.keyword:
            resp.extend(
                await vr.search_by_keyword(Chunk, query, self.search.keyword.topk)
            )
        if self.search.graph:
            resp.extend(await self.graph_search(query, Chunk, Entity, Relation, vr))
        resp.deduplicate()
        if self.rerank:
            if self.multimodal_emb:
                indices = await self.rerank.rerank_multimodal(
                    query=query,
                    chunks=[chunk.text for chunk in resp.chunks],
                    doc_type=resp.chunk_type,
                )
            else:
                indices = await self.rerank.rerank(
                    query=query, chunks=[chunk.text for chunk in resp.chunks]
                )
            resp.reorder(indices)
        if self.evaluate:
            resp.metrics = await self.evaluate.evaluate_with_estimation(
                query, [chunk.text for chunk in resp.chunks], chunk_type=resp.chunk_type
            )
        resp.cleanup()
        return resp

    async def graph_search(
        self,
        query: str,
        chunk_cls: type[Table],
        ent_cls: type[Table],
        rel_cls: type[Table],
        vr: VechordRegistry,
    ):
        ents, rels = await self.graph.recognize_with_relations(query)
        emb_func = (
            self.text_emb.vectorize_query
            if self.text_emb
            else self.multimodal_emb.vectorize_multimodal_query
        )
        if rels:
            rel_text = " ".join(rel.description for rel in rels)
            similar_rels = await vr.search_by_vector(
                rel_cls,
                await emb_func(rel_text),
                topk=self.search.graph.similar_k,
            )
            ent_uuids = deduplicate_uid(
                itertools.chain.from_iterable(
                    (rel.source, rel.target) for rel in similar_rels
                )
            )
            ents.extend(
                await vr.select_by(
                    ent_cls.partial_init(uid=AnyOf(ent_uuids)),
                    fields=("text", "description"),
                )
            )
        if not ents:
            return []
        ent_text = " ".join(f"{ent.text} {ent.description}" for ent in ents)
        similar_ents = await vr.search_by_vector(
            ent_cls,
            await emb_func(ent_text),
            topk=self.search.graph.similar_k,
        )
        chunk_uuids = deduplicate_uid(
            itertools.chain.from_iterable(ent.chunk_uuids for ent in similar_ents),
        )
        chunks = await vr.select_by(
            chunk_cls.partial_init(uid=AnyOf(chunk_uuids)),
            fields=("text", "doc_id", "uid"),
        )
        return chunks[: self.search.graph.topk]


def deduplicate_uid(uuids: Iterable[UUID], limit: Optional[int] = None) -> list[UUID]:
    """Maintain the order of the occurrence of UUIDs and deduplicate them."""
    uuids = {uid: None for uid in uuids}
    return list(uuids.keys())[:limit]
