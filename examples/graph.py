import csv
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import httpx
import msgspec
import rich.progress

from vechord.embedding import GeminiDenseEmbedding
from vechord.evaluate import BaseEvaluator
from vechord.graph import GeminiEntityRecognizer
from vechord.registry import VechordRegistry
from vechord.spec import AnyOf, PrimaryKeyUUID, Table, Vector

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DEFAULT_DATASET = "scifact"
TOP_K = 10

DenseVector = Vector[3072]
emb = GeminiDenseEmbedding()
ner = GeminiEntityRecognizer()


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
    vec: DenseVector


class Entity(Table, kw_only=True):
    uuid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    text: str
    label: str
    vec: DenseVector
    chunk_uuids: list[UUID]


class Relation(Table, kw_only=True):
    uuid: PrimaryKeyUUID = msgspec.field(default_factory=PrimaryKeyUUID.factory)
    source: UUID
    target: UUID
    text: str
    vec: DenseVector


class Evaluation(msgspec.Struct):
    map: float
    ndcg: float
    recall: float


vr = VechordRegistry(
    "graph",
    "postgresql://postgres:postgres@172.17.0.1:5432/",
    tables=[Chunk, Query, Entity, Relation],
)


@vr.inject(output=Chunk)
async def load_corpus(dataset: str, output: Path) -> AsyncIterator[Chunk]:
    file = output / dataset / "corpus.jsonl"
    decoder = msgspec.json.Decoder()
    entities: dict[str, Entity] = {}
    relations: dict[str, Relation] = {}
    with file.open("r") as f:
        for line in f:
            item = decoder.decode(line)
            text = f"{item['title']}\n{item['text']}"
            try:
                vector = await emb.vectorize_chunk(text)
            except Exception as e:
                print(f"failed to vectorize {text}: {e}")
                continue
            ents, rels = await ner.recognize_with_relations(text)
            for ent in ents:
                if ent.text not in entities:
                    entities[ent.text] = Entity(
                        text=ent.text,
                        label=ent.label,
                        vec=DenseVector(
                            await emb.vectorize_chunk(f"{ent.text} {ent.description}")
                        ),
                        chunk_uuids=[],
                    )
            for rel in rels:
                if rel.source.text not in entities or rel.target.text not in entities:
                    continue
                if rel.description not in relations:
                    relations[rel.description] = Relation(
                        source=entities[rel.source.text].uuid,
                        target=entities[rel.target.text].uuid,
                        text=rel.description,
                        vec=DenseVector(await emb.vectorize_chunk(rel.description)),
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
    await vr.copy_bulk(list(entities.values()))
    await vr.copy_bulk(list(relations.values()))


@vr.inject(output=Query)
async def load_query(dataset: str, output: Path) -> AsyncIterator[Query]:
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
                vec=DenseVector(await emb.vectorize_query(text)),
            )


async def expand_by_text(text: str) -> list[Chunk]:
    ents = await ner.recognize(text)
    chunks = []
    for ent in ents:
        entity = await vr.select_by(Entity.partial_init(text=ent.text))
        if not entity:
            continue
        entity = entity[0]
        chunks.extend(
            await vr.select_by(Chunk.partial_init(uuid=AnyOf(entity.chunk_uuids)))
        )
    return chunks


async def expand_by_graph(text: str, topk=3) -> list[Chunk]:
    ents, rels = await ner.recognize_with_relations(text)
    if not ents:
        return []
    entity_text = " ".join(f"{ent.text} {ent.description}" for ent in ents)
    similar_ents = await vr.search_by_vector(
        Entity, await emb.vectorize_query(entity_text), topk=topk
    )
    ent_uuids = set(ent.uuid for ent in similar_ents)
    if rels:
        relation_text = " ".join(rel.description for rel in rels)
        similar_rels = await vr.search_by_vector(
            Relation, await emb.vectorize_query(relation_text), topk=topk
        )
        ent_uuids |= set(rel.source for rel in similar_rels) | set(
            rel.target for rel in similar_rels
        )
    chunks = []
    retrieved_ents = await vr.select_by(Entity.partial_init(uuid=AnyOf(ent_uuids)))
    for ent in retrieved_ents:
        chunks.extend(
            await vr.select_by(Chunk.partial_init(uuid=AnyOf(ent.chunk_uuids)))
        )
    return chunks


@vr.inject(input=Query)
async def evaluate(cid: str, text: str, vec: DenseVector) -> Evaluation:
    chunks: list[Chunk] = await vr.search_by_vector(Chunk, vec, topk=TOP_K)
    expands = await expand_by_graph(text)
    final_chunks = list({chunk.uuid: chunk for chunk in chunks + expands}.values())
    # TODO: rerank
    score = BaseEvaluator.evaluate_one(cid, [doc.uid for doc in final_chunks])
    return Evaluation(
        map=score.get("map"),
        ndcg=score.get("ndcg"),
        recall=score.get(f"recall_{TOP_K}"),
    )


async def display_graph(save_to_file: bool = True):
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.Graph()
    rels: list[Relation] = await vr.select_by(Relation.partial_init())
    ent_table: dict[UUID, Entity] = {}
    for rel in rels:
        for uuid in (rel.source, rel.target):
            if uuid not in ent_table:
                ent = await vr.select_by(Entity.partial_init(uuid=uuid))
                if ent:
                    ent_table[uuid] = ent[0]

    ents = list(ent_table.values())
    edge_labels = {}
    for i, ent in enumerate(ents):
        graph.add_node(ent.uuid, label=ent.label, text=ent.text, index=i)
    for rel in rels:
        graph.add_edge(rel.source, rel.target, text=rel.text)
        edge_labels[(rel.source, rel.target)] = rel.text

    fig = plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2500,
        node_color="skyblue",
        alpha=0.7,
        edgecolors="black",
        linewidths=1,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=graph.edges(), width=1, alpha=0.5, edge_color="gray"
    )
    nx.draw_networkx_labels(graph, pos, font_size=10)
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5
    )
    plt.title("Entity-Relation Graph")
    plt.axis("off")
    plt.tight_layout()
    if save_to_file:
        with open("graph.png", "wb") as f:
            fig.savefig(f, format="png")
        return
    else:
        plt.show()


async def main():
    save_dir = Path("datasets")
    async with vr, emb, ner:
        await download_dataset(DEFAULT_DATASET, save_dir)

        await load_corpus(DEFAULT_DATASET, save_dir)
        await load_query(DEFAULT_DATASET, save_dir)

        res: list[Evaluation] = await evaluate()
        print("ndcg", sum(r.ndcg for r in res) / len(res))
        print("recall@10", sum(r.recall for r in res) / len(res))

        await display_graph()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
