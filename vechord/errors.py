class APIKeyUnsetError(Exception):
    """Raised when the API key is not set."""

    def __init__(self, name: str):
        self.message = (
            f"API key `{name}` is not set. Please set the environment variable."
        )
        super().__init__(self.message)


class HTTPCallError(Exception):
    def __init__(self, msg: str, status_code: int, err_msg: str):
        self.message = f"{msg} [{status_code}]: {err_msg}"
        super().__init__(self.message)


class DecodeStructuredOutputError(Exception):
    """Raised when decoding structured output fails."""
