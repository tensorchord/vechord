from typing import Optional

import falcon
import msgspec
from defspec import OpenAPI, RenderTemplate
from falcon import App, Request, Response

from vechord.log import logger
from vechord.registry import Table, VechordRegistry


def validate_request(spec: type[msgspec.Struct], req: Request, resp: Response):
    try:
        if req.method == "GET":
            request = msgspec.convert(req.params, spec)
        else:
            request = msgspec.json.decode(req.stream.read(), type=spec)
    except (msgspec.ValidationError, msgspec.DecodeError) as err:
        logger.info(
            "failed to decode the request '%s' body %s: %s", req.path, spec, err
        )
        resp.status = falcon.HTTP_422
        resp.text = f"Validation error: {err}"
        resp.content_type = falcon.MEDIA_TEXT
        return None
    return request


def uncaught_exception_handler(
    req: Request, resp: Response, exc: Exception, params: dict
):
    logger.warning(
        "exception from endpoint '%s'",
        req.path,
        exc_info=exc,
    )
    raise falcon.HTTPError(falcon.HTTP_500)


class HealthCheck:
    def on_get(self, req: Request, resp: Response):
        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = "Ok"


class TableResource:
    def __init__(self, table: type[Table], registry: VechordRegistry):
        self.table_cls = table
        self.registry = registry

    def on_get(self, req: Request, resp: Response):
        table = self.table_cls.partial_init(**req.params)
        rows = self.registry.select_by(obj=table)
        resp.data = msgspec.json.encode(rows)

    def on_post(self, req: Request, resp: Response):
        table: Optional[Table] = validate_request(self.table_cls, req, resp)
        if table is None:
            return

        self.registry.insert(table)
        resp.status = falcon.HTTP_201

    def on_delete(self, req: Request, resp: Response):
        table = self.table_cls.partial_init(**req.params)
        self.registry.remove_by(obj=table)


class PipelineResource:
    def __init__(self, registry: VechordRegistry):
        self.registry = registry
        self.decoder = msgspec.json.Decoder()

    def on_post(self, req: Request, resp: Response):
        json = self.decoder.decode(req.stream.read())
        if not isinstance(json, dict):
            raise falcon.HTTPBadRequest(
                title="Invalid request",
                description="Request must be a JSON Dict",
            )
        self.registry.run(**json)


class OpenAPIResource:
    def __init__(self, tables: list[Table]) -> None:
        self.openapi = OpenAPI()
        self.openapi.register_route("/", "get", summary="health check")
        self.openapi.register_route("/api/pipeline", "post", summary="run the pipeline")
        for table in tables:
            path = f"/api/table/{table.name()}"
            self.openapi.register_route(
                path,
                "get",
                "get the table with partial attributes",
                query_type=table,
            )
            self.openapi.register_route(
                path,
                "delete",
                "delete table records according to partial attributes",
                query_type=table,
            )
            self.openapi.register_route(
                path,
                "post",
                "insert a new record to the table",
                request_type=table,
                request_content_type="json",
            )
        self.spec = self.openapi.to_json()

    def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_JSON
        resp.data = self.spec


class OpenAPIRender:
    def __init__(self, spec_url: str, template: RenderTemplate) -> None:
        self.template = template.value.format(spec_url=spec_url)

    def on_get(self, req: Request, resp: Response):
        resp.content_type = falcon.MEDIA_HTML
        resp.text = self.template


def create_web_app(registry: VechordRegistry) -> App:
    """Create a `Falcon` WSGI application for the given registry.

    This includes the:
    - health check
    - table GET/POST/DELETE
    - pipeline POST
    - OpenAPI spec and Swagger UI
    """
    app = App()
    app.add_route("/", HealthCheck())
    for table in registry.tables:
        app.add_route(
            f"/api/table/{table.name()}",
            TableResource(table=table, registry=registry),
        )
    app.add_route("/api/pipeline", PipelineResource(registry))
    app.add_route("/openapi/spec.json", OpenAPIResource(registry.tables))
    app.add_route(
        "/openapi/swagger", OpenAPIRender("/openapi/spec.json", RenderTemplate.SWAGGER)
    )
    app.add_error_handler(Exception, uncaught_exception_handler)
    return app
