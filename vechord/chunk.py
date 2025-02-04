import re
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    @abstractmethod
    def segment(self, text: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class RegexChunker(BaseChunker):
    def __init__(
        self,
        size: int = 1536,
        overlap: int = 200,
        separator: str = r"\s{2,}",
        concat: str = ". ",
    ):
        self.size = size
        self.overlap = overlap
        self.separator = re.compile(separator)
        self.concatenator = concat

    def name(self) -> str:
        return f"regex_{self.size}_{self.overlap}"

    def keep_overlap(self, pieces: list[str]) -> list[str]:
        length = 0
        i = len(pieces) - 1
        while i >= 0:
            length += len(pieces[i])
            if length >= self.overlap:
                break
            i -= 1
        return pieces[i + 1 :]

    def segment(self, text: str) -> list[str]:
        chunks = []
        previous = []
        current = []
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
    def __init__(self):
        """A semantic sentence Chunker based on SpaCy."""
        import spacy

        self.nlp = spacy.load("en_core_web_sm", enable=["parser", "tok2vec"])

    def name(self) -> str:
        return "spacy"

    def segment(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]


class WordLlamaChunker(BaseChunker):
    def __init__(self, size: int = 1536):
        """A semantic chunker based on WordLlama."""
        from wordllama import WordLlama

        self.model = WordLlama.load()
        self.size = size

    def name(self) -> str:
        return f"wordllama_{self.size}"

    def segment(self, text: str) -> list[str]:
        return self.model.split(text, target_size=self.size)
