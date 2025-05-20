import csv
import zipfile
from pathlib import Path
from typing import Iterator
from uuid import UUID

import httpx
import msgspec
import rich.progress

from vechord.embedding import SpacyDenseEmbedding
from vechord.entity import SpacyEntityRecognizer
from vechord.evaluate import BaseEvaluator
from vechord.registry import VechordRegistry
from vechord.spec import PrimaryKeyUUID, Table, Vector

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DEFAULT_DATASET = "scifact"
TOP_K = 10

DenseVector = Vector[96]
emb = SpacyDenseEmbedding()
ner = SpacyEntityRecognizer()


def download_dataset(dataset: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    zip = output / f"{dataset}.zip"
    if not zip.is_file():
        with (
            zip.open("wb") as f,
            httpx.stream("GET", BASE_URL.format(dataset)) as stream,
        ):
            total = int(stream.headers["Content-Length"])
            with rich.progress.Progress(
                "[progress.percentage]{task.percentage:>3.0f}%",
                rich.progress.BarColumn(bar_width=None),
                rich.progress.DownloadColumn(),
                rich.progress.TransferSpeedColumn(),
            ) as progress:
                download_task = progress.add_task("Download", total=total)
                for chunk in stream.iter_bytes():
                    f.write(chunk)
                    progress.update(
                        download_task, completed=stream.num_bytes_downloaded
                    )
    unzip_dir = output / dataset
    if not unzip_dir.is_dir():
        with zipfile.ZipFile(zip, "r") as f:
            f.extractall(output)
    return unzip_dir


class Chunk(Table, kw_only=True):
    uuid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    uid: str
    text: str
    vec: DenseVector
    ent_uuids: list[UUID]


class Query(Table):
    uid: str
    cid: str
    text: str
    vector: DenseVector


class Entity(Table, kw_only=True):
    uuid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    label: str
    vec: DenseVector
    chunk_uuids: list[UUID]


class Evaluation(msgspec.Struct):
    map: float
    ndcg: float
    recall: float


vr = VechordRegistry("graph", "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Chunk, Query, Entity])


@vr.inject(output=Chunk)
def load_corpus(dataset: str, output: Path) -> Iterator[Chunk]:
    file = output / dataset / "corpus.jsonl"
    decoder = msgspec.json.Decoder()
    entities: dict[str, Entity] = {}
    with file.open("r") as f:
        for line in f:
            item = decoder.decode(line)
            text = f"{item['title']}\n{item['text']}"
            try:
                vector = emb.vectorize_chunk(text)
            except Exception as e:
                print(f"failed to vectorize {text}: {e}")
                continue
            ents = ner.predict(text)
            for ent in ents:
                if ent.text not in entities:
                    entities[ent.text] = Entity(
                        text=ent.text,
                        label=ent.label,
                        vec=DenseVector(emb.vectorize_chunk(ent.text)),
                        chunk_uuids=[],
                    )

            chunk = Chunk(
                uid=item["_id"],
                text=text,
                vec=DenseVector(vector),
                ent_uuids=[entities[ent.text].uuid for ent in ents],
            )
            for ent in ents:
                entities[ent.text].chunk_uuids.append(chunk.uuid)
            yield chunk
    vr.copy_bulk(list(entities.values()))


@vr.inject(output=Query)
def load_query(dataset: str, output: Path) -> Iterator[Query]:
    file = output / dataset / "queries.jsonl"
    truth = output / dataset / "qrels" / "test.tsv"

    table = {}
    with open(truth, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # skip header
        for row in reader:
            table[row[0]] = row[1]

    decoder = msgspec.json.Decoder()
    with file.open("r") as f:
        for line in f:
            item = decoder.decode(line)
            uid = item["_id"]
            if uid not in table:
                continue
            text = item.get("text", "")
            yield Query(
                uid=uid,
                cid=table[uid],
                text=text,
                vector=DenseVector(emb.vectorize_query(text)),
            )


def expand_by_text(text: str) -> list[Chunk]:
    ents = ner.predict(text)
    chunks = []
    for ent in ents:
        chunks.extend(
            res[0]
            for res in [
                vr.select_by(Chunk.partial_init(uuid=chunk_uuid))
                for chunk_uuid in ent.chunk_uuids
            ]
        )
    return chunks


def expand_by_similar_entity(entity: Entity, topk=3) -> list[Chunk]:
    ents = vr.search_by_vector(Entity, entity.vec, topk=topk)
    chunks = []
    for ent in ents:
        chunks.extend(
            res[0]
            for res in [
                vr.select_by(Chunk.partial_init(uuid=chunk_uuid))
                for chunk_uuid in ent.chunk_uuids
            ]
        )
    return chunks


def expand_by_chunk(chunk: Chunk) -> list[Chunk]:
    # TODO: add special SQL support
    # TODO: scores
    ents = [
        res[0]
        for res in [
            vr.select_by(Entity.partial_init(uuid=ent_uuid))
            for ent_uuid in chunk.ent_uuids
        ]
    ]
    chunks = []
    for ent in ents:
        chunks.extend(
            [
                res[0]
                for res in [
                    vr.select_by(Chunk.partial_init(uuid=chunk_uuid))
                    for chunk_uuid in ent.chunk_uuids
                ]
            ]
        )
    return chunks


@vr.inject(input=Query)
def evaluate(cid: str, text: str, vector: DenseVector) -> Evaluation:
    chunks: list[Chunk] = vr.search_by_vector(Chunk, vector, topk=TOP_K)
    expands = expand_by_text(text)
    final_chunks = list({chunk.uuid: chunk for chunk in chunks + expands}.values())
    # TODO: rerank
    score = BaseEvaluator.evaluate_one(cid, [doc.uid for doc in final_chunks])
    return Evaluation(
        map=score.get("map"),
        ndcg=score.get("ndcg"),
        recall=score.get(f"recall_{TOP_K}"),
    )


if __name__ == "__main__":
    save_dir = Path("datasets")
    download_dataset(DEFAULT_DATASET, save_dir)

    load_corpus(DEFAULT_DATASET, save_dir)
    load_query(DEFAULT_DATASET, save_dir)

    res: list[Evaluation] = evaluate()
    print("ndcg", sum(r.ndcg for r in res) / len(res))
    print("recall@10", sum(r.recall for r in res) / len(res))
