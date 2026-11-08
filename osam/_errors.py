# The errors osam reports to a caller, and how each one crosses HTTP: the
# server answers with the status and code in REPORTED, and the client raises
# the type in RESPONSE_ERRORS for that status, so a call raises the same
# builtin on a server as in process. An error not listed here is a 500, the
# one status after which the client restarts a private server.
from typing import Final
from typing import NamedTuple


class UnknownModelError(LookupError):
    # A model name osam doesn't know.
    pass


class UnknownImageError(LookupError):
    # Image references whose embeddings the runtime doesn't have.

    refs: list[str]

    def __init__(self, *, refs: list[str]):
        super().__init__(f"no embedding for {', '.join(refs)}; send the images")
        self.refs = refs


class BaseChangedError(ValueError):
    # An action whose base isn't the result of the actions before it.

    result: str

    def __init__(self, *, index: int, result: str):
        super().__init__(
            f"actions[{index}].base is not the result of the actions before it, "
            f"{result}; show it and make the edit again"
        )
        self.result = result


class DownloadError(RuntimeError):
    # A model file that no endpoint served.
    pass


class ResponseError(Exception):
    """An error the server returned, with its HTTP status code and its code,
    such as "model_not_found", which stays the same across releases."""

    status_code: int
    code: str | None

    def __init__(self, message: str, status_code: int, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class InvalidRequestError(ResponseError, ValueError):
    # A request the server rejected, such as a prompt that mixes one object
    # with every match: a ValueError, as in a local call.
    pass


class NotFoundError(ResponseError, LookupError):
    # Such as an unknown model: a LookupError, as in a local call.
    pass


class PullError(ResponseError, RuntimeError):
    # A download that failed on the server, in a pull or before a model
    # loads: a RuntimeError, as in a local call.
    pass


class Reported(NamedTuple):
    status_code: int
    code: str


# What the server did not expect, and so every error that REPORTED lacks.
INTERNAL: Final = Reported(status_code=500, code="internal_error")

# An error is reported as its own entry, or else its nearest base class's.
REPORTED: Final[dict[type[Exception], Reported]] = {
    UnknownModelError: Reported(status_code=404, code="model_not_found"),
    UnknownImageError: Reported(status_code=412, code="missing_images"),
    ValueError: Reported(status_code=400, code="invalid_request"),
    BaseChangedError: Reported(status_code=409, code="base_changed"),
    # 502, as the host of the model files failed, not osam's server.
    DownloadError: Reported(status_code=502, code="pull_failed"),
}

# By status code; any other status is a plain ResponseError. 422 is a request
# body the server could not validate.
RESPONSE_ERRORS: Final[dict[int, type[ResponseError]]] = {
    400: InvalidRequestError,
    404: NotFoundError,
    422: InvalidRequestError,
    502: PullError,
}
