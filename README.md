<div align="center">
<img src="https://github.com/user-attachments/assets/7b2819bb-1a7d-4b84-9ff9-d0c4d5340da9">

<p>

[![Python Check][ci-check-badge]][ci-check-file]
[![Pages][ci-page-badge]][document-link]
[![GitHub License][license-badge]][license-link]
[![PyPI - Version][pypi-badge]][pypi-link]
[![Discord][discord-badge]][discord-link]

</p>
<p><em>Turn PostgreSQL into your search engine in a Pythonic way.</em></p>
</div>

## Installation

```sh
pip install vechord
```

## Features

- [x] vector search with [RaBitQ][rabitq] (powered by [VectorChord][vectorchord])
- [x] multivec search with [WARP][xtr-warp] (powered by [VectorChord][vectorchord])
- [x] keyword search with BM25 score (powered by [VectorChord-bm25][vectorchord-bm25])
- [x] reduce boilerplate code by taking full advantage of the Python type hint
- [x] provide decorator to inject the data from/to the database
- [x] guarantee the data consistency with transaction
- [x] auto-generate the web service
- [x] provide common functions like (can also use any other libraries as you like):
  - [x] `Augmenter` for contextual retrieval
  - [x] `Chunker` to segment the text into chunks
  - [x] `Embedding` to generate the embedding from the text
  - [x] `Evaluator` to evaluate the search results with `NDCG`, `MAP`, `Recall`, etc.
  - [x] `Extractor` to extract the content from PDF, HTML, etc.
  - [x] `Reranker` for hybrid search

## Examples

- [simple.py](examples/simple.py): for people that are familiar with specialized vector database APIs
- [beir.py](examples/beir.py): the most flexible way to use the library (loading, indexing, querying and evaluation)
- [web.py](examples/web.py): build a web application with from the defined tables and pipeline
- [essay.py](examples/essay.py): extract the content from Paul Graham's essays and evaluate the search results from LLM generated queries
- [contextual.py](examples/contextual.py): contextual retrieval example
- [hybrid.py](examples/hybrid.py): hybrid search that rerank the results from vector search with keyword search

## User Guide

For the API references, check our [documentation][document-link].

### Define the table

```python
from typing import Annotated, Optional
from vechord.spec import Table, Vector, PrimaryKeyAutoIncrease, ForeignKey

# use 768 dimension vector
DenseVector = Vector[768]

class Document(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None  # auto-increase id, no need to set
    link: str = ""
    text: str

class Chunk(Table, kw_only=True)
    uid: Optional[PrimaryKeyAutoIncrease] = None
    doc_id: Annotated[int, ForeignKey[Document.uid]]  # reference to `Document.uid` on DELETE CASCADE
    vector: DenseVector  # this comes with a default vector index
    text: str
```

### Inject with decorator

```python
import httpx
from vechord.registry import VechordRegistry
from vechord.extract import SimpleExtractor
from vechord.embedding import GeminiDenseEmbedding

vr = VechordRegistry(namespace="test", url="postgresql://postgres:postgres@127.0.0.1:5432/")
# ensure the table and index are created if not exists
vr.register([Document, Chunk])
extractor = SimpleExtractor()
emb = GeminiDenseEmbedding()

@vr.inject(output=Document)  # dump to the `Document` table
# function parameters are free to define since `inject(input=...)` is not set
def add_document(url: str) -> Document:  # the return type is `Document`
    with httpx.Client() as client:
        resp = client.get(url)
        text = extractor.extract_html(resp.text)
        return Document(link=url, text=text)

@vr.inject(input=Document, output=Chunk)  # load from the `Document` table and dump to the `Chunk` table
# function parameters are the attributes of the `Document` table, only defined attributes
# will be loaded from the `Document` table
def add_chunk(uid: int, text: str) -> list[Chunk]:  # the return type is `list[Chunk]`
    chunks = text.split("\n")
    return [Chunk(doc_id=uid, vector=emb.vectorize_chunk(t), text=t) for t in chunks]

if __name__ == "__main__":
    add_document("https://paulgraham.com/best.html")  # add arguments as usual
    add_chunk()  # omit the arguments since the `input` is will be loaded from the `Document` table
    vr.insert(Document(text="hello world"))  # insert manually
    print(vr.select_by(Document.partial_init()))  # select all the columns from table `Document`
```

### Transaction

To guarantee the data consistency, users can use the `VechordRegistry.run` method to run multiple
functions in a transaction.

In this transaction, all the functions will only load the data from the database that is inserted
in the current transaction. So users can focus on the data processing part without worrying about
which part of data has not been processed yet.

```python
pipeline = vr.create_pipeline([add_document, add_chunk])
pipeline.run("https://paulgraham.com/best.html")  # only accept the arguments for the first function
```

### Search

```python
print(vr.search_by_vector(Chunk, emb.vectorize_query("startup")))
```

### Customized Index Configuration

```python
from vechord.spec import VectorIndex

class Chunk(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    vector: Annotated[DenseVector, VectorIndex(distance="cos", lists=128)]
    text: str
```

### Access the underlying database cursor directly

```python
vr.client.get_cursor().execute("SET vchordrq.probes = 100;")
```

### HTTP Service

This creates a WSGI application that can be served by any WSGI server.

Open the [OpenAPI Endpoint](http://127.0.0.1:8000/openapi/swagger) to check the API documentation.

```python
from vechord.service import create_web_app
from wsgiref.simple_server import make_server

app = create_web_app(vr)
with make_server("", 8000, app) as server:
    server.serve_forever()
```

## Development

```bash
docker run --rm -d --name vdb -e POSTGRES_PASSWORD=postgres -p 5432:5432 ghcr.io/tensorchord/vchord_bm25-postgres:pg17-v0.1.1
envd up
# inside the envd env, sync all the dependencies
make sync
# format the code
make format
```

[vectorchord]: https://github.com/tensorchord/VectorChord/
[vectorchord-bm25]: https://github.com/tensorchord/VectorChord-bm25
[rabitq]: https://github.com/gaoj0017/RaBitQ
[xtr-warp]:https://github.com/jlscheerer/xtr-warp
[ci-check-badge]: https://github.com/tensorchord/vechord/actions/workflows/check.yml/badge.svg
[ci-check-file]: https://github.com/tensorchord/vechord/actions/workflows/check.yml
[ci-page-badge]: https://github.com/tensorchord/vechord/actions/workflows/pages.yml/badge.svg
[document-link]: https://tensorchord.github.io/vechord/
[license-badge]: https://img.shields.io/github/license/tensorchord/vechord
[license-link]: https://github.com/tensorchord/vechord/blob/main/LICENSE
[pypi-badge]: https://img.shields.io/pypi/v/vechord
[pypi-link]: https://pypi.org/project/vechord/
[discord-badge]: https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60
[discord-link]: https://discord.gg/KqswhpVgdU
