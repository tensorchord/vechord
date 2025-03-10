from collections import defaultdict
from typing import Optional

from vechord.augment import BaseAugmenter
from vechord.chunk import BaseChunker
from vechord.client import VectorChordClient, hash_table_suffix
from vechord.embedding import BaseEmbedding
from vechord.evaluate import BaseEvaluator
from vechord.extract import BaseExtractor
from vechord.load import BaseLoader
from vechord.log import logger
from vechord.model import Chunk, Document, RetrievedChunk
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
            contexts = [
                Chunk(text=f"{context}\n{sent}", seq_id=i)
                for i, (context, sent) in enumerate(
                    zip(self.augmenter.augment_context(sentences), sentences)
                )
            ]
            queries = [
                Chunk(text=query, seq_id=i)
                for i, query in enumerate(self.augmenter.augment_query(sentences))
            ]
            summary = Chunk(text=self.augmenter.summarize_doc())
            chunks = [Chunk(text=sent, seq_id=i) for i, sent in enumerate(sentences)]
            self.client.insert_doc(doc, chunks + contexts + queries + [summary])
        else:
            chunks = [Chunk(text=sent) for sent in sentences]
            self.client.insert_doc(doc, chunks)

    def run(self):
        self.client.create()
        for doc in self.loader.load():
            if self.client.is_file_exists(doc):
                logger.debug("file %s already exists", doc.path)
                continue
            self.insert(doc)

    def clear(self):
        for doc in self.loader.load():
            self.client.delete_doc(doc)

    def query(self, query: str, topk: int = 10) -> list[RetrievedChunk]:
        resp = self.client.query_chunk(Chunk(text=query), topk)
        return resp

    def eval(self, evaluator: BaseEvaluator, topk: int = 10):
        res = []
        doc_ids = [row[0] for row in self.client.select("doc", ["id"])]
        all_chunks = self.client.select(
            f"chunk_{hash_table_suffix(self.identifier)}", ["id", "doc_id", "content"]
        )
        logger.debug(
            "got %d docs and %d chunks to evaluate", len(doc_ids), len(all_chunks)
        )
        for doc_id in doc_ids:
            chunks = [row[2] for row in all_chunks if row[1] == doc_id]
            ids = [row[0] for row in all_chunks if row[1] == doc_id]
            doc = "\n".join(chunks)
            queries = [evaluator.produce_query(doc, chunk) for chunk in chunks]
            retrieves = [self.query(query, topk) for query in queries]
            res.append(evaluator.evaluate(ids, retrieves))

        final = defaultdict(float)
        for r in res:
            for k, v in r.items():
                final[k] += v
        return {k: v / len(res) for k, v in final.items()}
