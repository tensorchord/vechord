<div align="center">
<img src="https://github.com/user-attachments/assets/7b2819bb-1a7d-4b84-9ff9-d0c4d5340da9">

<p>

[![Python Check][ci-check-badge]][ci-check-file]
[![Pages][ci-page-badge]][document-link]
[![GitHub License][license-badge]][license-link]
[![PyPI - Version][pypi-badge]][pypi-link]
[![Discord][discord-badge]][discord-link]
[![Blog][blog-badge]][blog-link]

</p>
<p><em>Turn PostgreSQL into your search engine in a Pythonic way.</em></p>
</div>

## Installation

```sh
pip install vechord
```

The related Docker images can be found in [VectorChord Suite][vectorchord-suite].

- DockerHub: `tensorchord/vchord-suite:pg17-20250620`
- GitHub Packages: `ghcr.io/tensorchord/vchord-suite:pg17-20250620`

## Features

- [x] vector search with [RaBitQ][rabitq] (powered by [VectorChord][vectorchord])
- [x] multivec search with [WARP][xtr-warp] (powered by [VectorChord][vectorchord])
- [x] keyword search with BM25 score (powered by [VectorChord-bm25][vectorchord-bm25])
- [x] reduce boilerplate code by taking full advantage of the Python type hint
- [x] provide decorator to inject the data from/to the database
- [x] guarantee the data consistency with the PostgreSQL transaction
- [x] auto-generate the web service
- [x] provide common tools like (can also use any other libraries):
  - [x] `Augmenter` for contextual retrieval
  - [x] `Chunker` to segment the text into chunks
  - [x] `Embedding` to generate the embedding from the text
  - [x] `Evaluator` to evaluate the search results with `NDCG`, `MAP`, `Recall`, etc.
  - [x] `Extractor` to extract the content from PDF, HTML, etc.
  - [x] `EntityRecognizer` to extract the entities and relations from the text
  - [x] `Reranker` for hybrid search

## Examples

- [simple.py](examples/simple.py): for people that are familiar with specialized vector database APIs
- [beir.py](examples/beir.py): the most flexible way to use the library (loading, indexing, querying and evaluation)
- [web.py](examples/web.py): build a web application with from the defined tables and pipeline
- [essay.py](examples/essay.py): extract the content from Paul Graham's essays and evaluate the search results from LLM generated queries
- [contextual.py](examples/contextual.py): contextual retrieval example with local PDF
- [anthropic.py](examples/anthropic.py): contextual retrieval with the Anthropic's Tutorial example
- [hybrid.py](examples/hybrid.py): hybrid search that rerank the results from vector search with keyword search
- [graph.py](examples/graph.py): graph-like entity-relation retrieval

## User Guide

For more details, check our [API reference][document-api] and [User Guide][document-guide].

### Define the table

```python
from typing import Annotated, Optional
from vechord.spec import Table, Vector, PrimaryKeyAutoIncrease, ForeignKey

# use 3072 dimension vector
DenseVector = Vector[3072]

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

vr = VechordRegistry(namespace="test", url="postgresql://postgres:postgres@127.0.0.1:5432/", tables=[Document, Chunk])
extractor = SimpleExtractor()
emb = GeminiDenseEmbedding()

@vr.inject(output=Document)  # dump to the `Document` table
# function parameters are free to define since `inject(input=...)` is not set
async def add_document(url: str) -> Document:  # the return type is `Document`
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        text = extractor.extract_html(resp.text)
        return Document(link=url, text=text)

@vr.inject(input=Document, output=Chunk)  # load from the `Document` table and dump to the `Chunk` table
# function parameters are the attributes of the `Document` table, only defined attributes
# will be loaded from the `Document` table
async def add_chunk(uid: int, text: str) -> list[Chunk]:  # the return type is `list[Chunk]`
    chunks = text.split("\n")
    return [Chunk(doc_id=uid, vector=await emb.vectorize_chunk(t), text=t) for t in chunks]

async def main():
    async with vr, emb:  # handle the connection with context manager
        await add_document("https://paulgraham.com/best.html")  # add arguments as usual
        await add_chunk()  # omit the arguments since the `input` is will be loaded from the `Document` table
        await vr.insert(Document(text="hello world"))  # insert manually
        print(await vr.select_by(Document.partial_init()))  # select all the columns from table `Document`

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Transaction

To guarantee the data consistency, users can use the `VechordRegistry.run` method to run multiple
functions in a transaction.

In this transaction, all the functions will only load the data from the database that is inserted
in the current transaction. So users can focus on the data processing part without worrying about
which part of data has not been processed yet.

```python
pipeline = vr.create_pipeline([add_document, add_chunk])
await pipeline.run("https://paulgraham.com/best.html")  # only accept the arguments for the first function
```

### Search

```python
print(await vr.search_by_vector(Chunk, await emb.vectorize_query("startup")))
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
await vr.client.get_cursor().execute("SET vchordrq.probes = 100;")
```

### HTTP Service

This creates a WSGI application that can be served by any WSGI server.

Open the [OpenAPI Endpoint](http://127.0.0.1:8000/openapi/swagger) to check the API documentation.

```python
import uvicorn

uvicorn.run(create_web_app(vr))
```

## Development

```bash
docker run --rm -d --name vdb -e POSTGRES_PASSWORD=postgres -p 5432:5432 ghcr.io/tensorchord/vchord-suite:pg17-20250620
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
[document-api]: https://tensorchord.github.io/vechord/api.html
[document-guide]: https://tensorchord.github.io/vechord/guide.html
[license-badge]: https://img.shields.io/github/license/tensorchord/vechord
[license-link]: https://github.com/tensorchord/vechord/blob/main/LICENSE
[pypi-badge]: https://img.shields.io/pypi/v/vechord
[pypi-link]: https://pypi.org/project/vechord/
[discord-badge]: https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60
[discord-link]: https://discord.gg/KqswhpVgdU
[vectorchord-suite]: https://github.com/tensorchord/VectorChord-images
[blog-badge]: https://img.shields.io/badge/VectorChrod-Blog-DAFDBA
[blog-link]: https://blog.vectorchord.ai/
