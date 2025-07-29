class VechordError(Exception):
    """Base class for all Vechord exceptions."""


class APIKeyUnsetError(VechordError):
    """Raised when the API key is not set."""

    def __init__(self, name: str):
        self.message = (
            f"API key `{name}` is not set. Please set the environment variable."
        )
        super().__init__(self.message)


class HTTPCallError(VechordError):
    def __init__(
        self, msg: str, status_code: int = 500, err_msg: str = "internal error"
    ):
        self.message = f"{msg} [{status_code}]: {err_msg}"
        super().__init__(self.message)

    def __str__(self):
        return f"Failed to call external LLM/Embedding services: {self.message}"


class DecodeStructuredOutputError(VechordError):
    """Raised when decoding structured output fails."""

    def __str__(self):
        return "Failed to decode structured output from LLM services"


class UnexpectedResponseError(VechordError):
    """Raised when the HTTP response is not as expected."""

    def __str__(self):
        return "Unexpected response from LLM/Embedding services"


class RequestError(VechordError):
    """Raised with bad request."""


class TimeoutError(VechordError):
    """Raised when a request times out."""


def extract_safe_err_msg(exc: Exception) -> str:
    if not isinstance(exc, VechordError):
        return ""
    return str(exc)
