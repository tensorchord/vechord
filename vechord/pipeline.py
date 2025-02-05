from typing import Optional

from vechord.augment import BaseAugmenter
from vechord.chunk import BaseChunker
from vechord.client import VectorChordClient
from vechord.embedding import BaseEmbedding
from vechord.extract import BaseExtractor
from vechord.load import BaseLoader
from vechord.log import logger
from vechord.model import Chunk, ChunkType, Document
from vechord.rerank import BaseReranker, ReciprocalRankFusion


class Pipeline:
    def __init__(  # noqa: PLR0913
        self,
        client: VectorChordClient,
        loader: BaseLoader,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        emb: BaseEmbedding | list[BaseEmbedding],
        rank_fusion: Optional[ReciprocalRankFusion] = None,
        augmenter: Optional[BaseAugmenter] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        self.client = client
        self.loader = loader
        self.extractor = extractor
        self.chunker = chunker
        self.augmenter = augmenter
        self.embs = emb if isinstance(emb, list) else [emb]
        self.reranker = reranker
        self.fusion = rank_fusion or ReciprocalRankFusion()

        # distinguish extractor & chunker
        self.identifier = f"{self.extractor.name()}_{self.chunker.name()}"
        self.client.set_context(self.identifier, self.embs, self.augmenter)

    def insert(self, doc: Document):
        text = self.extractor.extract(doc)
        sentences = self.chunker.segment(text)
        logger.debug("get %d chunks from doc %s", len(sentences), doc.path)
        if self.augmenter:
            self.augmenter.reset(text)
            contexts = self.augmenter.augment_context(sentences)
            queries = self.augmenter.augment_query(sentences)
            summary = self.augmenter.summarize_doc()
            chunks = (
                [Chunk(text=sent) for sent in sentences]
                + [Chunk(text=summary, chunk_type=ChunkType.SUMMARY)]
                + [Chunk(text=query, chunk_type=ChunkType.QUERY) for query in queries]
                + [
                    Chunk(text=f"{context}\n{sent}", chunk_type=ChunkType.CONTEXT)
                    for context, sent in zip(contexts, sentences)
                ]
            )
        chunks = [Chunk(text=sent) for sent in sentences]
        self.client.insert(doc, chunks)

    def run(self):
        self.client.create()
        doc_digests = []  # noqa  TODO: remove outdated docs
        for doc in self.loader.load():
            if self.client.is_file_exists(doc):
                logger.debug("file %s already exists", doc.path)
                continue
            self.insert(doc)

    def query(self, query: str) -> list[str]:
        resp = self.client.query(Chunk(text=query))
        return resp
