import os
from abc import ABC, abstractmethod
from datetime import timedelta


class BaseAugmenter(ABC):
    @abstractmethod
    def augment_context(self, chunks: list[str]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def augment_query(self, chunks: list[str]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def summarize_doc(self) -> str:
        raise NotImplementedError


class GeminiAugmenter(BaseAugmenter):
    def __init__(
        self, doc: str, model: str = "models/gemini-1.5-flash-001", ttl_sec: int = 600
    ):
        """Gemini Augmenter with cache."""
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("env GEMINI_API_KEY not set")

        import google.generativeai as genai

        cache = genai.caching.CachedContent.create(
            model=model,
            system_instruction=(
                "You are an expert on the natural language understanding. "
                "Answer the questions based on the whole document you have access to."
            ),
            contents=doc,
            ttl=timedelta(seconds=ttl_sec),
        )
        self.client = genai.GenerativeModel.from_cached_content(cached_content=cache)

    def augment(self, chunks: list[str], prompt: str) -> list[str]:
        res = []
        for chunk in chunks:
            response = self.client.generate_content([prompt.format(chunk=chunk)])
            res.append(response.text)
        return res

    def augment_chunks(self, chunks: list[str]) -> list[str]:
        prompt = (
            "Here is the chunk we want to situate within the whole document "
            "<chunk>{chunk}</chunk>"
            "Please give a short succinct context to situate this chunk within "
            "the overall document for the purposes of improving search retrieval "
            "of the chunk. Answer only with the succinct context and nothing else."
        )
        return self.augment(chunks, prompt)

    def augment_queries(self, chunks: list[str]) -> list[str]:
        prompt = (
            "Here is the chunk we want to ask questions about "
            "<chunk>{chunk}</chunk>"
            "Please ask questions about this chunk based on the overall document "
            "for the purposes of improving search retrieval of the chunk. "
            "Answer only with the question and nothing else."
        )
        return self.augment(chunks, prompt)

    def summarize_doc(self) -> str:
        prompt = (
            "Summarize the provided document concisely while preserving its key "
            "ideas, main arguments, and essential details. Ensure clarity and "
            "coherence, avoiding unnecessary repetition."
        )
        response = self.client.generate_content([prompt])
        return response.text
