import asyncio
from collections.abc import AsyncIterable, Iterable
from typing import Any, get_origin

import httpx

from vechord.typing import Self

# https://ai.google.dev/gemini-api/docs/rate-limits#tier-1
GEMINI_GENERATE_RPS = 16.66
GEMINI_EMBEDDING_RPS = 0.6
# https://docs.voyageai.com/docs/rate-limits
VOYAGE_EMBEDDING_RPS = 33.33
# https://jina.ai/api-dashboard/rate-limit
JINA_EMBEDDING_RPS = 8.33
JINA_RERANK_RPS = 8.33


def is_list_obj(obj) -> bool:
    """Check if the object is a list."""
    return isinstance(obj, (Iterable, AsyncIterable))


def is_list_of_type(typ) -> bool:
    """Check if the **type hint** is a list or iterable."""
    origin = get_origin(typ)
    if origin is None:
        return False
    return issubclass(origin, (Iterable, AsyncIterable))


def get_iterator_type(typ) -> type:
    if not is_list_of_type(typ):
        return typ
    return get_iterator_type(typ.__args__[0])


class RateLimitTransport(httpx.AsyncHTTPTransport):
    def __init__(self, max_per_second: float = 5, **kwargs) -> None:
        """
        Async HTTP transport with rate limit.

        Args:
            max_per_second: Maximum number of requests per second.

        Other args are passed to httpx.AsyncHTTPTransport.
        """
        self.interval = 1 / max_per_second
        self.next_start_time = 0
        super().__init__(**kwargs)

    async def notify_task_start(self):
        """
        https://github.com/florimondmanca/aiometer/blob/358976e0b60bce29b9fe8c59807fafbad3e62cbc/src/aiometer/_impl/meters.py#L57
        """
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            next_start_time = max(self.next_start_time, now)
            until_now = next_start_time - now
            if until_now <= self.interval:
                break
            await asyncio.sleep(max(0, until_now - self.interval))
        self.next_start_time = max(self.next_start_time, now) + self.interval

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self.notify_task_start()
        return await super().handle_async_request(request)

    async def __aenter__(self) -> Self:
        await self.notify_task_start()
        return await super().__aenter__()

    async def __aexit__(self, *args: Any) -> None:
        await super().__aexit__(*args)
