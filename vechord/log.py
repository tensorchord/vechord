import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = RichHandler()
logger.addHandler(handler)

# disable waitress log handler if any
logging.getLogger().addHandler(logging.NullHandler())
