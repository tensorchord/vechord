from abc import ABC, abstractmethod

import msgspec

from vechord.errors import DecodeStructuredOutputError
from vechord.model import (
    Entity,
    GeminiGenerateRequest,
    Relation,
)
from vechord.provider import GeminiGenerateProvider


class BaseEntityRecognizer(ABC):
    @abstractmethod
    def recognize(self, text: str) -> list[Entity]:
        """Predict the entities in the text."""
        raise NotImplementedError

    @abstractmethod
    def recognize_with_relations(
        self, text: str
    ) -> tuple[list[Entity], list[Relation]]:
        """Predict the entities and relations in the text."""
        raise NotImplementedError


class SpacyEntityRecognizer(BaseEntityRecognizer):
    """Spacy Entity Recognizer."""

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(model)
        self.matcher = spacy.matcher.Matcher(self.nlp.vocab)
        self.matcher.add(
            "ENT_VERB_ENT",
            [
                [
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                    {"POS": "VERB", "OP": "+"},
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                ]
            ],
        )
        self.matcher.add(
            "ENT_PREP_ENT",
            [
                [
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                    {"POS": "AUX", "OP": "*"},
                    {"POS": "VERB"},
                    {"POS": "ADP"},
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                ]
            ],
        )
        self.matcher.add(
            "ENT_POSSESSIVE_ENT",
            [
                [
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"LOWER": {"IN": ["'s"]}, "OP": "?"},
                    {"POS": "NOUN"},
                    {"LOWER": "is", "OP": "?"},
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                ]
            ],
        )
        self.matcher.add(
            "ENT_APPOSITION_ENT",
            [
                [
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"POS": "NOUN", "OP": "+"},
                    {"LOWER": "of", "OP": "?"},
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                ]
            ],
        )
        self.matcher.add(
            "ENT_ATTRIBUTE_ENT",
            [
                [
                    {"ENT_TYPE": {"NOT_IN": [""]}},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"POS": "NOUN"},
                    {"LIKE_NUM": True},
                ]
            ],
        )
        self.model = model

    def recognize(self, text) -> list[Entity]:
        doc = self.nlp(text)
        return [Entity(text=ent.text, label=ent.label_) for ent in doc.ents]

    def recognize_with_relations(self, text) -> tuple[list[Entity], list[Relation]]:
        doc = self.nlp(text)
        ents = [Entity(text=ent.text, label=ent.label_) for ent in doc.ents]
        relations: list[Relation] = []
        matches = self.matcher(doc)
        for _, start, end in matches:
            span = doc[start:end]
            ent0 = ent1 = None
            for token in span:
                if token.ent_type_:
                    ent = Entity(text=token.text, label=token.ent_type_)
                    if ent0 is None:
                        ent0 = ent
                    else:
                        ent1 = ent

            relations.append(
                Relation(
                    source=ent0 or Entity(text=span[0].text, label=span[0].ent_type_),
                    target=ent1 or Entity(text=span[-1].text, label=span[-1].ent_type_),
                    description=" ".join(token.text for token in span),
                )
            )

        return ents, relations


class GeminiEntityRecognizer(BaseEntityRecognizer, GeminiGenerateProvider):
    """Entity recognizer using Gemini API.

    Since Gemini SDK doesn't support passing JSON schema directly, we have to
    build the HTTP request manually. Otherwise, we can only use pydantic models
    to define the returned types, which is not ideal for this use case. Internally,
    Gemini also generates the JSON schema from the pydantic model.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)

    async def recognize(self, text) -> list[Entity]:
        prompt = (
            "Given the text document, identify and extract all the entities, return the JSON "
            "format with the following fields: "
            "- text: the entity text "
            "- label: the entity type (e.g., PER, ORG, LOC, TIME, GPE, VEH etc.) "
            "- description: a brief description of the entity in the current context "
            "\n<document>\n{text}\n</document>\n"
        )
        resp = await self.query(
            GeminiGenerateRequest.from_prompt_structure_response(
                prompt=prompt.format(text), schema=msgspec.json.schema(list[Entity])
            )
        )
        try:
            ents = msgspec.json.decode(resp.get_text(), type=list[Entity])
        except (msgspec.DecodeError, KeyError) as err:
            raise DecodeStructuredOutputError(
                "Failed to decode Gemini response"
            ) from err
        return ents

    async def recognize_with_relations(
        self, text
    ) -> tuple[list[Entity], list[Relation]]:
        prompt = (
            "Given the text document, extract entities and the possible relations "
            "between them. Return a list of relations with source and target "
            "entities in JSON format like: "
            "- source: the source entity (text, label and description) "
            "- target: the target entity (text, label and description) "
            "- description: a brief description of the relation in the current context "
            "\n<document>\n{text}\n</document>\n"
        )
        resp = await self.query(
            GeminiGenerateRequest.from_prompt_structure_response(
                prompt=prompt.format(text=text),
                schema=msgspec.json.schema(list[Relation]),
            )
        )
        try:
            rels = msgspec.json.decode(resp.get_text(), type=list[Relation])
        except (msgspec.DecodeError, KeyError) as err:
            raise DecodeStructuredOutputError(
                "Failed to decode Gemini response"
            ) from err
        ents: dict[str, Entity] = {}
        for rel in rels:
            ents[rel.source.text] = rel.source
            ents[rel.target.text] = rel.target
        return list(ents.values()), rels


if __name__ == "__main__":

    async def main():
        async with GeminiEntityRecognizer() as recognizer:
            text = "Barack Obama was the 44th President of the United States. He was born in Hawaii. His wife, Michelle Obama, is a lawyer and writer."

            ents, rels = await recognizer.recognize_with_relations(text)
            print(ents)
            print(rels)

    import asyncio

    asyncio.run(main())
