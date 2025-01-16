import numpy as np
import spacy
from openai import OpenAI
from spacy.tokens import Doc


class TextProcessor:
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model, enable=["tagger", "parser", "ner", "tok2vec"])

    def process(self, content: str) -> Doc:
        return self.nlp(content)


EN_TEXT_PROCESSOR = TextProcessor()


class TextEmbeddingClient:
    def __init__(self, api_key: str, api_url: str, model: str, timeout: float = 30):
        assert len(api_key) > 0, "API key cannot be empty"
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)

    def embedding(self, text: str) -> np.ndarray:
        return np.array(
            self.client.embeddings.create(model=self.model, input=text)
            .data[0]
            .embedding
        )
