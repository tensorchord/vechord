from argparse import ArgumentParser
from typing import Optional

import falcon
import msgspec
from falcon import App, Request, Response

from vechord.log import logger
from vechord.model import Chunk, Document
from vechord.pipeline import Pipeline


def build_parser():
    parser = ArgumentParser(prog="vechord")
    parser.add_argument("--debug", action="store_true", help="enable debug log")
    parser.add_argument(
        "--host", type=str, default="localhost", help="host of the server"
    )
    parser.add_argument("--port", type=int, default=8000, help="port of the server")
    return parser


def validate_request(spec: type[msgspec.Struct], req: Request, resp: Response):
    buf = req.stream.read()
    try:
        request = msgspec.json.decode(buf, type=spec)
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


class DocumentResource:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def on_post(self, req: Request, resp: Response):
        doc: Optional[Document] = validate_request(Document, req, resp)
        if doc is None:
            return
        self.pipeline.insert(doc=doc)
        resp.status_code = falcon.HTTP_201

    def on_delete(self, req: Request, resp: Response):
        doc: Optional[Document] = validate_request(Document, req, resp)
        if doc is None:
            return
        self.pipeline.client.delete(doc=doc)


class QueryResource:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def on_post(self, req: Request, resp: Response):
        query: Optional[Chunk] = validate_request(Chunk, req, resp)
        if query is None:
            return
        res = self.pipeline.query(query=query.text)
        resp.data = msgspec.json.encode(res)


def create_web_app(pipeline: Pipeline) -> App:
    app = App()
    app.add_route("/health", HealthCheck())
    app.add_route("/document", DocumentResource(pipeline))
    app.add_route("/query", QueryResource(pipeline))
    app.add_error_handler(Exception, uncaught_exception_handler)
    return app


if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    from vechord.chunk import SpacyChunker
    from vechord.client import VectorChordClient
    from vechord.embedding import SpacyDenseEmbedding
    from vechord.load import LocalLoader

    parser = build_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel("DEBUG")
    logger.debug(args)

    pipeline = Pipeline(
        client=VectorChordClient(
            "test", "postgresql://postgres:postgres@172.17.0.1:5432/"
        ),
        loader=LocalLoader("data"),
        chunker=SpacyChunker(),
        emb=SpacyDenseEmbedding(),
    )
    web_app = create_web_app(pipeline)
    with make_server(args.host, args.port, web_app) as server:
        logger.info("serving on %s:%d", args.host, args.port)
        server.serve_forever()
