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
    def __init__(self, msg: str, status_code: int, err_msg: str):
        self.message = f"{msg} [{status_code}]: {err_msg}"
        super().__init__(self.message)


class DecodeStructuredOutputError(VechordError):
    """Raised when decoding structured output fails."""


class UnexpectedResponseError(VechordError):
    """Raised when the HTTP response is not as expected."""


class RequestError(VechordError):
    """Raised with bad request."""


def extract_safe_err_msg(exc: Exception) -> str:
    if not isinstance(exc, VechordError):
        return ""

    if isinstance(exc, HTTPCallError):
        return "Failed to call external LLM/Embedding services"
    elif isinstance(exc, DecodeStructuredOutputError):
        return "Failed to decode structured output from LLM services"
    elif isinstance(exc, UnexpectedResponseError):
        return "Unexpected response from LLM/Embedding services"
    elif isinstance(exc, (RequestError, APIKeyUnsetError)):
        return exc.message
