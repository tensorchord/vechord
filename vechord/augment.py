from abc import ABC, abstractmethod

from vechord.model import GeminiGenerateRequest
from vechord.provider import GeminiGenerateProvider


class BaseAugmenter(ABC):
    @abstractmethod
    def reset(self, doc: str):
        """Cache the document for augmentation."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def augment_context(self, doc: str, chunks: list[str]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def augment_query(self, doc: str, chunks: list[str]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def summarize_doc(self, doc: str) -> str:
        raise NotImplementedError


class GeminiAugmenter(BaseAugmenter, GeminiGenerateProvider):
    """Gemini Augmenter.

    Context caching is only available for stable models with fixed versions.
    Minimal cache token is 32768.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)

    def name(self) -> str:
        return f"gemini_augment_{self.model}"

    async def augment(self, doc: str, chunks: list[str], prompt: str) -> list[str]:
        res = []
        for chunk in chunks:
            context = prompt.format(chunk=chunk)
            context = f"<document>\n{doc}\n</document>\n" + context
            resp = await self.query(GeminiGenerateRequest.from_prompt(context))
            res.append(resp.get_text().strip())
        return res

    async def augment_context(self, doc: str, chunks: list[str]) -> list[str]:
        """Generate the contextual chunks.

        Args:
            doc: document text
            chunks: list of chunks to augment
        """
        prompt = (
            "Here is the chunk we want to situate within the whole document \n"
            "<chunk>\n{chunk}\n</chunk>\n"
            "Please give a short succinct context to situate this chunk within "
            "the overall document for the purposes of improving search retrieval "
            "of the chunk. Answer only with the succinct context and nothing else."
        )
        return await self.augment(doc, chunks, prompt)

    async def augment_query(self, doc: str, chunks: list[str]) -> list[str]:
        """Generate the queries for chunks.

        Args:
            doc: document text
            chunks: list of chunks to augment
        """
        prompt = (
            "Here is the chunk we want to ask questions about \n"
            "<chunk>\n{chunk}\n</chunk>\n"
            "Please ask questions about this chunk based on the overall document "
            "for the purposes of improving search retrieval of the chunk. "
            "Answer only with the question and nothing else."
        )
        return await self.augment(doc, chunks, prompt)

    async def summarize_doc(self, doc: str) -> str:
        """Summarize the document."""
        prompt = (
            "Summarize the provided document concisely while preserving its key "
            "ideas, main arguments, and essential details. Ensure clarity and "
            "coherence, avoiding unnecessary repetition."
            f"\n<document>{doc}</document>\n"
        )
        resp = await self.query(GeminiGenerateRequest.from_prompt(prompt))
        return resp.get_text().strip()
