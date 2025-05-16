from abc import ABC, abstractmethod

from vechord.model import Entity


class BaseEntityRecognizer(ABC):
    @abstractmethod
    def predict(self, text: str) -> list[Entity]:
        """Predict the entities in the text."""
        raise NotImplementedError


class SpacyEntityRecognizer(BaseEntityRecognizer):
    """Spacy Entity Recognizer."""

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(model, enable=["ner"])
        self.model = model

    def predict(self, text) -> list[Entity]:
        doc = self.nlp(text)
        return [Entity(text=ent.text, label=ent.label_) for ent in doc.ents]
