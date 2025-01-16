from argparse import ArgumentParser

from vechord.log import logger


def build_parser():
    parser = ArgumentParser(prog="vechord")
    parser.add_argument("--debug", action="store_true", help="enable debug log")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logger.debug(args)
