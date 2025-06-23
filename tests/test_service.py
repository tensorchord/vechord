from http import HTTPStatus

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
    def create_chunk(uid: int, text: str) -> list[DefaultChunk]:
        nums = [int(x) for x in text.split()]
        return [
            DefaultChunk(
                doc_id=uid,
                text=f"num[{num}]",
                keyword=Keyword(num),
                vec=gen_vector(),
            )
            for num in nums
        ]

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
