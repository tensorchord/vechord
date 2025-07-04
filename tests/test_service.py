import asyncio
from collections.abc import AsyncIterator
from http import HTTPStatus
from uuid import UUID

import msgspec
import pytest
from falcon import testing

from tests.conftest import gen_vector
from vechord.service import create_web_app
from vechord.spec import DefaultDocument, Keyword, create_chunk_with_dim

pytestmark = pytest.mark.anyio
DefaultChunk = create_chunk_with_dim(128)


@pytest.fixture(name="client")
async def fixture_client(registry):
    @registry.inject(output=DefaultDocument)
    def create_doc(text: str) -> DefaultDocument:
        return DefaultDocument(text=text)

    @registry.inject(input=DefaultDocument, output=DefaultChunk)
    async def create_chunk(uid: UUID, text: str) -> AsyncIterator[DefaultChunk]:
        nums = [int(x) for x in text.split()]
        for num in nums:
            yield DefaultChunk(
                doc_id=uid,
                text=f"num[{num}]",
                keyword=Keyword(num),
                vec=gen_vector(),
            )

    app = create_web_app(registry, registry.create_pipeline([create_doc, create_chunk]))

    async with testing.TestClient(app) as client:
        yield client


@pytest.mark.parametrize("registry", [(DefaultDocument, DefaultChunk)], indirect=True)
async def test_service_health(client):
    resp = await client.simulate_get("/")
    assert resp.status_code == HTTPStatus.OK


@pytest.mark.parametrize("registry", [(DefaultDocument, DefaultChunk)], indirect=True)
async def test_service_openapi(client):
    resp = await client.simulate_get("/openapi/spec.json")
    assert resp.status_code == HTTPStatus.OK
    assert resp.json is not None

    resp = await client.simulate_get("/openapi/swagger")
    assert resp.status_code == HTTPStatus.OK


@pytest.mark.parametrize("registry", [(DefaultDocument, DefaultChunk)], indirect=True)
async def test_service_table(client):
    resp = await client.simulate_get("/api/table/defaultdocument")
    assert resp.status_code == HTTPStatus.OK
    assert len(resp.json) == 0

    text = "hello world"

    doc = DefaultDocument(text=text)
    resp = await client.simulate_post(
        "/api/table/defaultdocument", body=msgspec.json.encode(doc)
    )
    assert resp.status_code == HTTPStatus.CREATED

    resp = await client.simulate_get("/api/table/defaultdocument")
    assert resp.status_code == HTTPStatus.OK
    assert len(resp.json) == 1
    assert resp.json[0]["text"] == text

    resp = await client.simulate_delete(
        "/api/table/defaultdocument", json={"text": text}
    )
    assert resp.status_code == HTTPStatus.OK

    resp = await client.simulate_get("/api/table/defaultdocument")
    assert resp.status_code == HTTPStatus.OK
    assert len(resp.json) == 0


@pytest.mark.parametrize("registry", [(DefaultDocument, DefaultChunk)], indirect=True)
async def test_service_pipeline(client):
    text = "1 2 3 4 5"
    resp = await client.simulate_post("/api/pipeline", json={"text": text})
    assert resp.status_code == HTTPStatus.OK

    resp = await client.simulate_get("/api/table/defaultchunk")
    assert resp.status_code == HTTPStatus.OK
    chunks = resp.json
    assert len(chunks) == len(text.split())
    for i, chunk in enumerate(chunks):
        assert chunk["text"] == f"num[{i + 1}]"


@pytest.mark.parametrize("registry", [(DefaultDocument, DefaultChunk)], indirect=True)
async def test_concurrent_db_transaction(client):
    requests = [
        client.simulate_post(
            "/api/pipeline", json={"text": " ".join(map(str, range(i, i + 5)))}
        )
        for i in range(5)
    ]
    responses = await asyncio.gather(*requests)
    for resp in responses:
        assert resp.status_code == HTTPStatus.OK, resp.content
