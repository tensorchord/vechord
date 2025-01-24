from vechord.client import VectorChordClient
from vechord.embedding import BaseEmbedding
from vechord.extract import BaseExtractor
from vechord.load import BaseLoader
from vechord.log import logger
from vechord.model import Chunk
from vechord.segment import BaseSegmenter


class Pipeline:
    def __init__(
        self,
        client: VectorChordClient,
        loader: BaseLoader,
        extractor: BaseExtractor,
        segmenter: BaseSegmenter,
        emb: BaseEmbedding,
    ):
        self.client = client
        self.loader = loader
        self.extractor = extractor
        self.segmenter = segmenter
        self.emb = emb

    def run(self):
        self.client.create(self.emb.get_dim())
        for file in self.loader.load():
            if self.client.is_file_exists(file):
                logger.debug("file %s already exists", file.path)
                continue
            text = self.extractor.extract(file)
            sentences = self.segmenter.segment(text)
            chunks = [
                Chunk(text=sent, vector=self.emb.vectorize(sent)) for sent in sentences
            ]
            self.client.insert_text(file, chunks)

    def query(self, query: str) -> list[str]:
        resp = self.client.query(Chunk(text=query, vector=self.emb.vectorize(query)))
        return resp
