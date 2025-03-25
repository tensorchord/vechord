<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="200" height="128" fill="none" viewBox="0 0 200 206">
<defs><path id="a" stroke="#EAB711" d="M0-8h40"/></defs>
<path stroke="#EAB711" stroke-width="16" d="M8 6v200M0 8h40M0 198h40M192 6v200"/>
<use xlink:href="#a" stroke-width="16" transform="matrix(-1 0 0 1 200 16)"/>
<use xlink:href="#a" stroke-width="16" transform="matrix(-1 0 0 1 200 206)"/>
<path fill="#3776AB" d="m75.91 67.91 22.5 70.726h.863l22.545-70.727h21.818L111.545 161H86.182L54.045 67.91z"/>
</svg>

<p>

[![Python Check](https://github.com/tensorchord/vechord/actions/workflows/check.yml/badge.svg)](https://github.com/tensorchord/vechord/actions/workflows/check.yml)
[![Pages](https://github.com/tensorchord/vechord/actions/workflows/pages.yml/badge.svg)]( tensorchord.github.io/vechord/)
![GitHub License](https://img.shields.io/github/license/tensorchord/vechord)
![PyPI - Version](https://img.shields.io/pypi/v/vechord)
[![Discord](https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60)](https://discord.gg/KqswhpVgdU)

</p>
<p><em>Turn PostgreSQL into your search engine in a Pythonic way.</em></p>
</div>

## Installation

```sh
pip install vechord
```

## Examples

- [beir.py](examples/beir.py): the most flexible way to use the library (loading, indexing, querying and evaluation)
- [web.py](examples/web.py): build a web application with from the defined tables and pipeline
- [essay.py](examples/essay.py): extract the content from Paul Graham's essays and evaluate the search results from LLM generated queries
- [contextual.py](examples/contextual.py): contextual retrieval example

## Development

```bash
docker run --rm -d --name vdb -e POSTGRES_PASSWORD=postgres -p 5432:5432 ghcr.io/tensorchord/vchord_bm25-postgres:pg17-v0.1.1
envd up
# inside the envd env, sync all the dependencies
make sync
# format the code
make format
```
