from vechord.augment import BaseAugmenter
from vechord.chunk import BaseChunker
from vechord.client import VectorChordClient
from vechord.embedding import BaseEmbedding
from vechord.extract import BaseExtractor
from vechord.load import BaseLoader
from vechord.log import logger
from vechord.model import Chunk
from vechord.rerank import BaseReranker, ReciprocalRankFusion


class Pipeline:
    def __init__(  # noqa: PLR0913
        self,
        client: VectorChordClient,
        loader: BaseLoader,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        emb: BaseEmbedding | list[BaseEmbedding],
        rank_fusion: ReciprocalRankFusion,
        augmenter: BaseAugmenter | None = None,
        reranker: BaseReranker | None = None,
    ):
        self.client = client
        self.loader = loader
        self.extractor = extractor
        self.chunker = chunker
        self.augmenter = augmenter
        self.embs = emb if isinstance(emb, list) else [emb]
        self.reranker = reranker
        self.fusion = rank_fusion

        # distinguish extractor & chunker
        self.identifier = f"{self.extractor.name()}_{self.chunker.name()}"
        self.client.set_context(self.identifier, self.embs)

    def run(self):
        self.client.create()
        for doc in self.loader.load():
            if self.client.is_file_exists(doc):
                logger.debug("file %s already exists", doc.path)
                continue
            text = self.extractor.extract(doc)
            sentences = self.chunker.segment(text)
            chunks = [Chunk(text=sent) for sent in sentences]
            self.client.insert_text(doc, chunks)

    def query(self, query: str) -> list[str]:
        resp = self.client.query(Chunk(text=query))
        return resp
