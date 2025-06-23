import os
from abc import ABC, abstractmethod

import httpx
import msgspec


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


class GeminiAugmenter(BaseAugmenter):
    """Gemini Augmenter.

    Context caching is only available for stable models with fixed versions.
    Minimal cache token is 32768.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("env GEMINI_API_KEY not set")

        self.model = model
        self.min_token = 32768
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self.client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()

    def name(self) -> str:
        return f"gemini_augment_{self.model}"

    async def augment(self, doc: str, chunks: list[str], prompt: str) -> list[str]:
        res = []
        for chunk in chunks:
            context = prompt.format(chunk=chunk)
            context = f"<document>\n{doc}\n</document>\n" + context
            resp = await self.client.post(
                url=self.url,
                json={"contents": [{"parts": [{"text": context}]}]},
                params={"key": self.api_key},
            )
            if resp.is_error:
                raise RuntimeError(f"Failed to augment with Gemini: {resp.text}")
            data = msgspec.json.decode(resp.content)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            res.append(text.strip())
        return res

    async def augment_context(self, doc: str, chunks: list[str]) -> list[str]:
        """Generate the contextual chunks."""
        prompt = (
            "Here is the chunk we want to situate within the whole document \n"
            "<chunk>\n{chunk}\n</chunk>\n"
            "Please give a short succinct context to situate this chunk within "
            "the overall document for the purposes of improving search retrieval "
            "of the chunk. Answer only with the succinct context and nothing else."
        )
        return await self.augment(doc, chunks, prompt)

    async def augment_query(self, doc: str, chunks: list[str]) -> list[str]:
        """Generate the queries for chunks."""
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
        resp = await self.client.post(
            url=self.url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            params={"key": self.api_key},
        )
        if resp.is_error:
            raise RuntimeError(f"Failed to augment with Gemini: {resp.text}")
        data = msgspec.json.decode(resp.content)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
