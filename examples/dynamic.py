"""
This can also be run as a service like:

```sh
vechord --db postgresql://postgres:postgres@127.0.0.1:5432
```

And send the request to the endpoint `POST /api/run`.
"""

from os import environ
from pathlib import Path
from uuid import UUID

from tqdm import tqdm

from vechord.model import InputType, ResourceRequest, RunRequest
from vechord.pipeline import DynamicPipeline
from vechord.registry import VechordRegistry

VOYAGE_API_KEY = environ.get("VOYAGE_API_KEY")
GEMINI_API_KEY = environ.get("GEMINI_API_KEY")
namespace = "dynamic"

vr = VechordRegistry(
    namespace=namespace, url="postgresql://postgres:postgres@172.17.0.1:5432/"
)
ingest_steps = [
    ResourceRequest(
        kind="multimodal-emb", provider="voyage", args={"api_key": VOYAGE_API_KEY}
    ),
    ResourceRequest(kind="index", provider="vectorchord", args={"vector": {}}),
]
search_steps = [
    ResourceRequest(
        kind="multimodal-emb", provider="voyage", args={"api_key": VOYAGE_API_KEY}
    ),
    ResourceRequest(
        kind="search", provider="vectorchord", args={"vector": {"topk": 10}}
    ),
]
file_uuids: dict[UUID, Path] = {}


async def ingest(files: list[Path]):
    dp = DynamicPipeline.from_steps(ingest_steps)
    for file in tqdm(files):
        ack = await dp.run(
            request=RunRequest(
                name=namespace, data=file.read_bytes(), input_type=InputType.IMAGE
            ),
            vr=vr,
        )
        file_uuids[ack.uid] = file


async def search(query: str):
    dp = DynamicPipeline.from_steps(search_steps)
    return await dp.run(
        request=RunRequest(name=namespace, data=query.encode("utf-8")),
        vr=vr,
    )


async def main():
    async with vr:
        dir = Path.home() / "Pictures"
        await ingest(dir.glob("*.jpg"))
        res = await search("cat")
        for item in res:
            print("=>", file_uuids.get(item.doc_id, "Unknown file"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
