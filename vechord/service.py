from typing import Any, Optional, TypeVar

import falcon
import msgspec
import numpy as np
from defspec import OpenAPI, RenderTemplate
from falcon.asgi import App, Request, Response

from vechord.errors import extract_safe_err_msg
from vechord.log import logger
from vechord.model import RunAck, RunRequest
from vechord.pipeline import DynamicPipeline
from vechord.registry import Table, VechordPipeline, VechordRegistry
from vechord.spec import _DefaultChunk

T = TypeVar("T")
M = TypeVar("M", bound=msgspec.Struct)


def vechord_schema_hook(cls: type):
    if method := getattr(cls, "__json_schema__", None):
        return method()
    raise NotImplementedError()


def vechord_encode_hook(obj: Any) -> Any:
    if method := getattr(obj, "__json_encode__", None):
        return method(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise NotImplementedError(
        f"Object {obj} does not implement __json_encode__ method."
    )


def vechord_decode_hook(obj_type: type[T], data: Any) -> T:
    if method := getattr(obj_type, "__json_decode__", None):
        return method(data)
    raise NotImplementedError(
        f"Type {obj_type} does not implement __json_decode__ method."
    )


async def validate_request(spec: type[M], req: Request, resp: Response) -> Optional[M]:
    decoder = (
        msgspec.msgpack if req.content_type == falcon.MEDIA_MSGPACK else msgspec.json
    )
    try:
        if req.method == "GET":
            request = msgspec.convert(req.params, spec)
        else:
            request = decoder.decode(
                await req.stream.read(), type=spec, dec_hook=vechord_decode_hook
            )
    except (msgspec.ValidationError, msgspec.DecodeError) as err:
        logger.info(
            "failed to decode the request '%s' body %s: %s", req.path, spec, err
        )
        resp.status = falcon.HTTP_422
        resp.text = f"Validation error: {err}"
        resp.content_type = falcon.MEDIA_TEXT
        return None
    return request


async def uncaught_exception_handler(
    req: Request, resp: Response, exc: Exception, params: dict
):
    logger.warning(
        "exception from endpoint '%s'",
        req.path,
        exc_info=exc,
    )
    description = extract_safe_err_msg(exc)
    raise falcon.HTTPError(falcon.HTTP_500, title=req.path, description=description)


class HealthCheck:
    async def on_get(self, req: Request, resp: Response):
        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = "Ok"


class TableResource:
    def __init__(self, table: type[Table], registry: VechordRegistry):
        self.table_cls = table
        self.registry = registry

    async def on_get(self, req: Request, resp: Response):
        table = self.table_cls.partial_init(**req.params)
        rows = await self.registry.select_by(obj=table)
        resp.data = msgspec.json.encode(rows, enc_hook=vechord_encode_hook)

    async def on_post(self, req: Request, resp: Response):
        table = await validate_request(self.table_cls, req, resp)
        if table is None:
            return

        await self.registry.insert(table)
        resp.status = falcon.HTTP_CREATED

    async def on_delete(self, req: Request, resp: Response):
        table = self.table_cls.partial_init(**req.params)
        await self.registry.remove_by(obj=table)


class PipelineResource:
    def __init__(self, pipeline: VechordPipeline):
        self.pipeline = pipeline
        self.decoder = msgspec.json.Decoder(dec_hook=vechord_decode_hook)

    async def on_post(self, req: Request, resp: Response):
        json = self.decoder.decode(await req.stream.read())
        if not isinstance(json, dict):
            raise falcon.HTTPBadRequest(
                title="Invalid request",
                description="Request must be a JSON Dict",
            )
        await self.pipeline.run(**json)


class RunResource:
    def __init__(self, registry: VechordRegistry):
        self.registry = registry

    async def on_post(self, req: Request, resp: Response):
        request = await validate_request(RunRequest, req, resp)
        if request is None:
            return
        pipe = DynamicPipeline.from_steps(request.steps)
        res = await pipe.run(request, self.registry)
        if res:
            encoder = (
                msgspec.msgpack
                if req.content_type == falcon.MEDIA_MSGPACK
                else msgspec.json
            )
            resp.data = encoder.encode(res, enc_hook=vechord_encode_hook)


class OpenAPIResource:
    def __init__(
        self, tables: list[type[Table]], include_pipeline: bool = True
    ) -> None:
        self.openapi = OpenAPI()
        self.openapi.register_route("/", "get", summary="health check")
        self.openapi.register_route(
            "/api/run",
            "post",
            summary="run the pipeline",
            request_type=RunRequest,
            response_type=RunAck | list[_DefaultChunk],
            schema_hook=vechord_schema_hook,
        )
        if include_pipeline:
            self.openapi.register_route(
                "/api/pipeline", "post", summary="run the pipeline"
            )
        for table in tables:
            path = f"/api/table/{table.name()}"
            self.openapi.register_route(
                path,
                "get",
                "get the table with partial attributes",
                query_type=table,
                schema_hook=vechord_schema_hook,
            )
            self.openapi.register_route(
                path,
                "delete",
                "delete table records according to partial attributes",
                query_type=table,
                schema_hook=vechord_schema_hook,
            )
            self.openapi.register_route(
                path,
                "post",
                "insert a new record to the table",
                request_type=table,
                request_content_type="json",
                schema_hook=vechord_schema_hook,
            )
        self.spec = self.openapi.to_json()

    async def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_JSON
        resp.data = self.spec


class OpenAPIRender:
    def __init__(self, spec_url: str, template: RenderTemplate) -> None:
        self.template = template.value.format(spec_url=spec_url)

    async def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_HTML
        resp.text = self.template


def create_web_app(
    registry: VechordRegistry, pipeline: Optional[VechordPipeline] = None
) -> App:
    """Create a `Falcon` ASGI application for the given registry.

    This includes the:

    - health check [GET](/)
    - tables [GET/POST/DELETE](/api/table/{table_name})
    - OpenAPI spec and Swagger UI [GET](/openapi/swagger)
    - optional: pipeline in a transaction [POST](/api/pipeline)
    """
    app = App(middleware=registry)
    app.req_options.strip_url_path_trailing_slash = True
    app.add_route("/", HealthCheck())
    for table in registry.tables:
        app.add_route(
            f"/api/table/{table.name()}",
            TableResource(table=table, registry=registry),
        )
    if pipeline is not None:
        app.add_route("/api/pipeline", PipelineResource(pipeline=pipeline))
    app.add_route("/api/run", RunResource(registry=registry))
    app.add_route(
        "/openapi/spec.json", OpenAPIResource(registry.tables, pipeline is not None)
    )
    app.add_route(
        "/openapi/swagger", OpenAPIRender("/openapi/spec.json", RenderTemplate.SWAGGER)
    )
    app.add_error_handler(Exception, uncaught_exception_handler)
    return app
