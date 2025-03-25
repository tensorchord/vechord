<div align="center">
<img src="https://github.com/user-attachments/assets/7b2819bb-1a7d-4b84-9ff9-d0c4d5340da9">

<p>

[![Python Check][ci-check-badge]][ci-check-file]
[![Pages][ci-page-badge]][document-link]
![GitHub License][license-badge]
![PyPI - Version][pypi-badge]
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
- [x] guarantee the data consistency with transaction (use the `VechordRegistry.run`)
- [x] provide decorator to inject the data from/to the database
- [x] auto-generate the web service

## Examples

- [simple.py](examples/simple.py): for people that are familiar with specialized vector database APIs
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
[rabitq]: https://github.com/gaoj0017/RaBitQ
[xtr-warp]:https://github.com/jlscheerer/xtr-warp
[ci-check-badge]: https://github.com/tensorchord/vechord/actions/workflows/check.yml/badge.svg
[ci-check-file]: https://github.com/tensorchord/vechord/actions/workflows/check.yml
[ci-page-badge]: https://github.com/tensorchord/vechord/actions/workflows/pages.yml/badge.svg
[document-link]: https://tensorchord.github.io/vechord/
[license-badge]: https://img.shields.io/github/license/tensorchord/vechord
[pypi-badge]: https://img.shields.io/pypi/v/vechord
[discord-badge]: https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60
[discord-link]: https://discord.gg/KqswhpVgdU
