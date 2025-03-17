import csv
import zipfile
from pathlib import Path
from typing import Iterator

import httpx
import msgspec
import rich.progress

from vechord.embedding import GeminiDenseEmbedding
from vechord.evaluate import BaseEvaluator
from vechord.registry import VechordRegistry
from vechord.spec import Table, Vector

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DEFAULT_DATASET = "scifact"
TOP_K = 10

emb = GeminiDenseEmbedding()
DenseVector = Vector[768]


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


class Corpus(Table):
    uid: str
    text: str
    title: str
    vector: DenseVector


class Query(Table):
    uid: str
    cid: str
    text: str
    vector: DenseVector


class Evaluation(msgspec.Struct):
    map: float
    ndcg: float
    recall: float


vr = VechordRegistry(DEFAULT_DATASET, "postgresql://postgres:postgres@172.17.0.1:5432/")
vr.register([Corpus, Query])


@vr.inject(output=Corpus)
def load_corpus(dataset: str, output: Path) -> Iterator[Corpus]:
    file = output / dataset / "corpus.jsonl"
    decoder = msgspec.json.Decoder()
    with file.open("r") as f:
        for line in f:
            item = decoder.decode(line)
            title = item.get("title", "")
            text = item.get("text", "")
            try:
                vector = emb.vectorize_chunk(f"{title}\n{text}")
            except Exception as e:
                print(f"failed to vectorize {title}: {e}")
                continue
            yield Corpus(
                uid=item["_id"],
                text=text,
                title=title,
                vector=vector,
            )


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
                uid=uid, cid=table[uid], text=text, vector=emb.vectorize_query(text)
            )


@vr.inject(input=Query)
def evaluate(cid: str, vector: DenseVector) -> Evaluation:
    docs: list[Corpus] = vr.search(Corpus, vector, topk=TOP_K)
    score = BaseEvaluator.evaluate_one(cid, [doc.uid for doc in docs])
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
