import os
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedding(ABC):
    @abstractmethod
    def vectorize(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError


class SpacyEmbedding(BaseEmbedding):
    def __init__(self, model: str = "en_core_web_sm", dim: int = 96):
        import spacy

        self.nlp = spacy.load(model, enable=["tok2vec"])
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vectorize(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        return doc.vector


class GeminiEmbedding(BaseEmbedding):
    def __init__(self, model: str = "models/text-embedding-004", dim: int = 768):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")

        import google.generativeai as genai

        self.client = genai.embed_content
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vectorize(self, text: str) -> np.ndarray:
        res = self.client(
            content=text, model=self.model, output_dimensionality=self.dim
        )
        return np.array(res.embedding)


class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model: str = "text-embedding-3-large", dim: int = 3072):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")

        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def embedding(self, text: str) -> np.ndarray:
        return np.array(
            self.client.embeddings.create(
                model=self.model, input=text, dimensions=self.dim
            )
            .data[0]
            .embedding
        )
