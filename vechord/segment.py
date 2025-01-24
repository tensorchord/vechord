import re
from abc import ABC, abstractmethod


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, text: str) -> list[str]:
        raise NotImplementedError


class RegexSegmenter(BaseSegmenter):
    def __init__(self, size: int, overlap: int, separator: str):
        self.size = size
        self.overlap = overlap
        self.separator = re.compile(separator)
        self.concatenator = ". "

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


class SpacySegmenter(BaseSegmenter):
    def __init__(self):
        import spacy

        self.nlp = spacy.load("en_core_web_sm", enable=["parser", "tok2vec"])

    def segment(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]
