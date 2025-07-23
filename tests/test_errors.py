from vechord.errors import (
    APIKeyUnsetError,
    DecodeStructuredOutputError,
    HTTPCallError,
    RequestError,
    UnexpectedResponseError,
    extract_safe_err_msg,
)


def test_extracted_msg():
    # other errors should return empty string
    assert extract_safe_err_msg(ValueError("what happened")) == ""

    # hide the real error message
    for exc in [
        DecodeStructuredOutputError,
        UnexpectedResponseError,
        HTTPCallError,
    ]:
        assert extract_safe_err_msg(exc("some error")) == str(exc("doesn't matter"))

    assert "UNIVERSAL" in extract_safe_err_msg(APIKeyUnsetError("UNIVERSAL"))
    assert "leaked" in extract_safe_err_msg(RequestError("leaked"))
