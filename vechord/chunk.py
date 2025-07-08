import os
import re
from abc import ABC, abstractmethod

import httpx
import msgspec

from vechord.model import GeminiGenerateRequest, GeminiGenerateResponse
from vechord.utils import GEMINI_GENERATE_RPS, RateLimitTransport


class BaseChunker(ABC):
    @abstractmethod
    async def segment(self, text: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class RegexChunker(BaseChunker):
    """A simple regex-based chunker."""

    def __init__(
        self,
        size: int = 1536,
        overlap: int = 200,
        separator: str = r"[\n\r\f\v\t?!.;]{1,}",
        concat: str = ". ",
    ):
        self.size = size
        self.overlap = overlap
        self.separator = re.compile(separator)
        self.concatenator = concat

    def name(self) -> str:
        return f"regex_chunk_{self.size}_{self.overlap}"

    def keep_overlap(self, pieces: list[str]) -> list[str]:
        length = 0
        i = len(pieces) - 1
        while i >= 0:
            length += len(pieces[i])
            if length >= self.overlap:
                break
            i -= 1
        return pieces[i + 1 :]

    async def segment(self, text: str) -> list[str]:
        chunks: list[str] = []
        previous: list[str] = []
        current: list[str] = []
        total_length = 0
        pieces = self.separator.split(text)

        for raw_piece in pieces:
            piece = raw_piece.strip()
            if not piece:
                continue
            if total_length + len(piece) > self.size and current:
                chunks.append(self.concatenator.join(previous + current))
                previous = self.keep_overlap(current)
                total_length = sum(len(p) for p in previous)
                current = []
            current.append(piece)
            total_length += len(piece)

            # the new added piece is too long
            if total_length > self.size:
                # pop the overlapping pieces
                overlap_index = 0
                while overlap_index < len(previous):
                    total_length -= len(previous[overlap_index])
                    if total_length <= self.size:
                        break
                    overlap_index += 1
                previous = previous[overlap_index + 1 :]
                chunks.append(self.concatenator.join(previous + current))
                previous = self.keep_overlap(current)
                total_length = sum(len(p) for p in previous)
                current = []

        remaining = self.concatenator.join(previous + current)
        return [*chunks, remaining] if remaining else chunks


class SpacyChunker(BaseChunker):
    """A semantic sentence Chunker based on SpaCy.

    This guarantees the generated chunks are sentences.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy

        self.model = model
        self.nlp = spacy.load(model, enable=["parser", "tok2vec"])

    def name(self) -> str:
        return f"spacy_chunk_{self.model}"

    async def segment(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]


class GeminiChunker(BaseChunker):
    """A semantic chunker based on Gemini."""

    def __init__(self, model: str = "gemini-2.5-flash", size: int = 1536):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("env GEMINI_API_KEY not set")

        self.model = model
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            timeout=httpx.Timeout(120.0, connect=5.0),
            transport=RateLimitTransport(max_per_second=GEMINI_GENERATE_RPS),
        )
        self.prompt = f"""
You are an expert text chunker, skilled at dividing documents into meaningful 
segments while respecting token limits. Your goal is to break down a document into 
chunks that are as semantically coherent as possible, ensuring no chunk exceeds a 
specified token length. Maintain document order. The maximum token length is {size}.
The return format is a list of chunk strings.The document is as follows:
"""
        self.output_token_limit = 65536
        self.regex_chunker = RegexChunker(
            size=self.output_token_limit,
            overlap=0,
            separator=r"[\n\r\f\v\t?!.;]{1,}",
            concat=" ",
        )
        self.json_schema = msgspec.json.schema(list[str])
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(GeminiGenerateResponse)

    def name(self) -> str:
        return f"gemini_chunk_{self.model}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        await self.client.aclose()

    async def query(self, prompt: str) -> list[str]:
        resp = await self.client.post(
            url=self.url,
            content=self.encoder.encode(
                GeminiGenerateRequest.from_prompt_structure_response(
                    prompt=prompt,
                    schema=self.json_schema,
                )
            ),
        )
        if resp.is_error:
            raise RuntimeError(f"Failed to chunk with Gemini: {resp.text}")
        data = self.decoder.decode(resp.content)
        chunks = msgspec.json.decode(data.get_text(), type=list[str])
        return chunks

    async def segment(self, text: str) -> list[str]:
        tokens = len(text)
        if tokens <= self.output_token_limit:
            chunks = await self.query(self.prompt + f"\n<document> {text} </document>")
            return chunks

        all_chunks = []
        for chunk in self.regex_chunker.segment(text):
            chunks = await self.query(self.prompt + f"\n<document> {chunk} </document>")
            all_chunks.extend(chunks)
        return all_chunks
