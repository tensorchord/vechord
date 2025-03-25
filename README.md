# Vechord

Python RAG framework built on top of PostgreSQL and [VectorChord](https://github.com/tensorchord/VectorChord/).

## Installation

```sh
pip install vechord
```

## Features

- [x] vector search with RaBitQ (powered by [VectorChord][vectorchord])
- [x] multivec search (powered by [VectorChord][vectorchord])
- [x] keyworx search with BM25 ranking (powered by [VectorChord-bm25][vectorchord-bm25])

## Examples

- [beir.py](examples/beir.py): the most flexible way to use the library (loading, indexing, querying and evaluation)
- [web.py](examples/web.py): build a web application with from the defined tables and pipeline
- [essay.py](examples/essay.py): extract the content from Paul Graham's essays and evaluate the search results from LLM generated queries
- [contextual.py](examples/contextual.py): contextual retrieval example
- [hybrid.py](examples/hybrid.py): hybrid search that rerank the results from vector search with keyword search

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
