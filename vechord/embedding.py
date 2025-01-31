import os
from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np


class EmbeddingType(Enum):
    DENSE = auto()
    SPARSE = auto()
    KEYWORD = auto()


class BaseEmbedding(ABC):
    emb_type = EmbeddingType.DENSE

    @abstractmethod
    def vectorize_doc(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def vectorize_query(self, text: str) -> np.ndarray:
        return self.vectorize_doc(text)

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError


class SpacyDenseEmbedding(BaseEmbedding):
    emb_type = EmbeddingType.DENSE

    def __init__(self, model: str = "en_core_web_sm", dim: int = 96):
        import spacy

        self.nlp = spacy.load(model, enable=["tok2vec"])
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vectorize_doc(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        return doc.vector


class GeminiDenseEmbedding(BaseEmbedding):
    emb_type = EmbeddingType.DENSE

    def __init__(self, model: str = "models/text-embedding-004", dim: int = 768):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("env GEMINI_API_KEY not set")

        import google.generativeai as genai

        self.client = genai.embed_content
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vectorize_doc(self, text: str) -> np.ndarray:
        res = self.client(
            content=text, model=self.model, output_dimensionality=self.dim
        )
        return np.array(res["embedding"])


class OpenAIDenseEmbedding(BaseEmbedding):
    emb_type = EmbeddingType.DENSE

    def __init__(self, model: str = "text-embedding-3-large", dim: int = 3072):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("env OPENAI_API_KEY not set")

        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def vectorize_doc(self, text: str) -> np.ndarray:
        return np.array(
            self.client.embeddings.create(
                model=self.model, input=text, dimensions=self.dim
            )
            .data[0]
            .embedding
        )
