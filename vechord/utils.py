import asyncio
import sys
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


import httpx


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
