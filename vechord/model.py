import hashlib
from pathlib import Path
from sys import version_info

import msgspec
from numpy import ndarray
from spacy.tokens import Span

from vechord.loader import DEFAULT_EXTRACTOR
from vechord.text import EN_TEXT_PROCESSOR

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class Entity(msgspec.Struct, kw_only=True, frozen=True):
    text: str
    label: str


class Sentence(msgspec.Struct, kw_only=True):
    text: str
    vector: ndarray
    entities: list[Entity] = []

    @classmethod
    def from_spacy_sentence(cls, span: Span) -> Self:
        return cls(
            text=span.text.replace("\x00", "-").replace("\n", " ").strip(),
            vector=span.vector,
            entities=[Entity(text=ent.text, label=ent.label_) for ent in span.ents],
        )


class TextFile(msgspec.Struct, kw_only=True):
    digest: str
    filename: str
    sentences: list[Sentence] = []

    @classmethod
    def from_filepath(cls, filename: Path) -> Self:
        stream = filename.read_bytes()
        digest = hashlib.sha256(stream).hexdigest()
        content = DEFAULT_EXTRACTOR[filename.suffix.lower()](stream)
        doc = EN_TEXT_PROCESSOR.process(content)
        sentences = [Sentence.from_spacy_sentence(sent) for sent in doc.sents]
        return cls(digest=digest, filename=str(filename), sentences=sentences)
