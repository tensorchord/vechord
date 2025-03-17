import os
from abc import ABC, abstractmethod
from datetime import timedelta

from vechord.log import logger


class BaseAugmenter(ABC):
    @abstractmethod
    def reset(self, doc: str):
        """Cache the document for augmentation."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

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
    """Gemini Augmenter.

    Context caching is only available for stable models with fixed versions.
    Minimal cache token is 32768.
    """

    def __init__(self, model: str = "models/gemini-1.5-flash-001", ttl_sec: int = 600):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("env GEMINI_API_KEY not set")

        self.model_name = model
        self.ttl_sec = ttl_sec
        self.min_token = 32768

    def name(self) -> str:
        return f"gemini_augment_{self.model_name}"

    def reset(self, doc: str):
        """Reset the document."""
        import google.generativeai as genai

        self.client = genai.GenerativeModel(model_name=self.model_name)
        tokens = self.client.count_tokens(doc).total_tokens
        self.doc = ""  # empty means doc is in the cache
        if tokens <= self.min_token:
            # cannot use cache due to the Gemini token limit
            self.doc = doc
        else:
            logger.debug("use cache since the doc has %d tokens", tokens)
            cache = genai.caching.CachedContent.create(
                model=self.model_name,
                system_instruction=(
                    "You are an expert on the natural language understanding. "
                    "Answer the questions based on the whole document you have access to."
                ),
                contents=doc,
                ttl=timedelta(seconds=self.ttl_sec),
            )
            self.client = genai.GenerativeModel.from_cached_content(
                cached_content=cache
            )

    def augment(self, chunks: list[str], prompt: str) -> list[str]:
        res = []
        try:
            for chunk in chunks:
                context = prompt.format(chunk=chunk)
                if self.doc:
                    context = f"<document>{self.doc}</document>\n" + context
                response = self.client.generate_content([context])
                res.append(response.text)
        except Exception as e:
            logger.error("GeminiAugmenter error: %s", e)
            breakpoint()
        return res

    def augment_context(self, chunks: list[str]) -> list[str]:
        """Generate the contextual chunks."""
        prompt = (
            "Here is the chunk we want to situate within the whole document "
            "<chunk>{chunk}</chunk>"
            "Please give a short succinct context to situate this chunk within "
            "the overall document for the purposes of improving search retrieval "
            "of the chunk. Answer only with the succinct context and nothing else."
        )
        return self.augment(chunks, prompt)

    def augment_query(self, chunks: list[str]) -> list[str]:
        """Generate the queries for chunks."""
        prompt = (
            "Here is the chunk we want to ask questions about "
            "<chunk>{chunk}</chunk>"
            "Please ask questions about this chunk based on the overall document "
            "for the purposes of improving search retrieval of the chunk. "
            "Answer only with the question and nothing else."
        )
        return self.augment(chunks, prompt)

    def summarize_doc(self) -> str:
        """Summarize the document."""
        prompt = (
            "Summarize the provided document concisely while preserving its key "
            "ideas, main arguments, and essential details. Ensure clarity and "
            "coherence, avoiding unnecessary repetition."
        )
        if self.doc:
            prompt = f"<document>{self.doc}</document>\n" + prompt
        response = self.client.generate_content([prompt])
        return response.text
