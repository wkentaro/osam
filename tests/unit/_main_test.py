import types
import typing
from typing import Any

from osam import __main__


def test_full_name_comes_from_the_server_list() -> None:
    models = [types.SimpleNamespace(name=n) for n in ["sam3:latest", "sam2:tiny"]]
    client = typing.cast(Any, types.SimpleNamespace(list=lambda: models))
    assert __main__._full_name(client=client, name="sam3") == "sam3:latest"
    assert __main__._full_name(client=client, name="sam2:tiny") == "sam2:tiny"
    # A name the server doesn't list stays as given.
    assert __main__._full_name(client=client, name="sam2") == "sam2"
