from typing import Optional

from vechord.embedding import GeminiDenseEmbedding
from vechord.registry import VechordRegistry
from vechord.spec import PrimaryKeyAutoIncrease, Table, Vector

DenseVector = Vector[768]


class Document(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    title: str = ""
    text: str
    vec: DenseVector


if __name__ == "__main__":
    vr = VechordRegistry("simple", "postgresql://postgres:postgres@172.17.0.1:5432/")
    vr.register([Document])
    emb = GeminiDenseEmbedding()

    # add a document
    text = "my personal long note"
    doc = Document(title="note", text=text, vec=DenseVector(emb.vectorize_chunk(text)))
    vr.insert(doc)

    # load
    docs = vr.select_by(Document.partial_init(), limit=1)
    print(docs)

    # query
    res = vr.search_by_vector(Document, emb.vectorize_query("note"), topk=1)
    print(res)
