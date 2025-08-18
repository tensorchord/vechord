from os import environ

import numpy as np
import pytest

from vechord.client import VechordClient
from vechord.log import logger
from vechord.registry import VechordRegistry
from vechord.spec import Vector


@pytest.fixture
def anyio_backend():
    # trio is not supported by psycopg
    # https://github.com/psycopg/psycopg/issues/29
    return "asyncio"


URL = "127.0.0.1"
# for local container development environment, use the host machine's IP
if environ.get("REMOTE_CONTAINERS", "") == "true" or environ.get("USER", "") == "envd":
    URL = "172.17.0.1"
TEST_POSTGRES = f"postgresql://postgres:postgres@{URL}:5432/"


@pytest.fixture(name="registry")
async def fixture_registry(request):
    namespace = request.node.obj.__name__
    tables = getattr(request, "param", ())
    async with VechordRegistry(namespace, TEST_POSTGRES, tables=tables) as registry:
        yield registry

    # when the registry is used as Falcon middleware, the lifespan protocol will close
    # the connection, so we need to re-create the database connection to cleanup
    logger.debug("clearing storage...")
    async with VechordClient(namespace, TEST_POSTGRES) as client:
        for table in tables:
            await client.drop(table.name())


DenseVector = Vector[128]


def gen_vector(dim: int = 128) -> DenseVector:
    rng = np.random.default_rng()
    return DenseVector(rng.random((dim,), dtype=np.float32))
