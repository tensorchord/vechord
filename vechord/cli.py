from argparse import ArgumentParser

import uvicorn

from vechord.registry import VechordRegistry
from vechord.service import create_web_app


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run vechord as a dynamic pipeline service.")
    parser.add_argument(
        "--db",
        type=str,
        default="postgresql://postgres:postgres@localhost:5432/",
        help="VectorChord DB URL.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Service host.")
    parser.add_argument("--port", type=int, default=8000, help="Service port.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    vr = VechordRegistry("cli", args.db)
    app = create_web_app(vr)
    uvicorn.run(app, host=args.host, port=args.port)
