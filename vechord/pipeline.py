from vechord.chunk import BaseChunker
from vechord.client import VectorChordClient
from vechord.embedding import BaseEmbedding
from vechord.extract import BaseExtractor
from vechord.load import BaseLoader
from vechord.log import logger
from vechord.model import Chunk
from vechord.augment import BaseAugmenter


class Pipeline:
    def __init__(
        self,
        client: VectorChordClient,
        loader: BaseLoader,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        augmenter: BaseAugmenter,
        emb: BaseEmbedding,
    ):
        self.client = client
        self.loader = loader
        self.extractor = extractor
        self.chunker = chunker
        self.augmenter = augmenter
        self.emb = emb

    def run(self):
        self.client.create(self.emb.get_dim())
        for doc in self.loader.load():
            if self.client.is_file_exists(doc):
                logger.debug("file %s already exists", doc.path)
                continue
            text = self.extractor.extract(doc)
            sentences = self.chunker.segment(text)
            chunks = [
                Chunk(text=sent, vector=self.emb.vectorize_doc(sent))
                for sent in sentences
            ]
            self.client.insert_text(doc, chunks)

    def query(self, query: str) -> list[str]:
        resp = self.client.query(
            Chunk(text=query, vector=self.emb.vectorize_query(query))
        )
        return resp
