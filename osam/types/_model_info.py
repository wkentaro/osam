import typing

import pydantic

from ._model_metadata import ModelMetadata

PromptKind = typing.Literal["points", "box", "texts", "example_boxes"]
InputResize = typing.Literal["stretch", "fit"]


def fit_hw(*, height: int, width: int, size: int) -> tuple[int, int]:
    if width > height:
        return max(1, round(height * size / width)), size
    return size, max(1, round(width * size / height))


class ModelInfo(pydantic.BaseModel):
    name: str
    id: str
    size: int | None = pydantic.Field(description="Bytes on disk; null if not pulled")
    download_size: int | None = pydantic.Field(
        default=None,
        description="Bytes a pull downloads for the device, pulled or not; "
        "null if unknown",
    )
    modified_at: float | None = pydantic.Field(
        description="Unix time of the last download; null if not pulled"
    )
    metadata: ModelMetadata
    capabilities: list[typing.Literal["segment", "track"]] = pydantic.Field(
        description="The endpoints the model serves: segment for /api/segment "
        "and track for /api/track"
    )
    prompt_kinds: list[PromptKind] = pydantic.Field(
        description="The prompt fields it takes; a request with others gets 400"
    )
    input_size: int = pydantic.Field(
        description="Side, in pixels, of the square the model resizes every image into"
    )
    input_resize: InputResize = pydantic.Field(
        description="How it resizes an image into the square: stretch to "
        "input_size by input_size, ignoring the aspect ratio, or fit the longer "
        "side to input_size, keeping the aspect ratio, and pad the rest"
    )

    def input_shape(self, *, height: int, width: int) -> tuple[int, int]:
        """The (height, width) the model resizes an image of this size to.

        An image that already has this shape reaches the image encoder unresized.
        """
        if self.input_resize == "stretch":
            return self.input_size, self.input_size
        return fit_hw(height=height, width=width, size=self.input_size)


class RunningModel(pydantic.BaseModel):
    name: str
    device: str = pydantic.Field(
        description='Device it runs on, such as "cuda", or "cpu" after a fallback'
    )
    expires_at: float | None = pydantic.Field(
        description="Unix time the model unloads; null if it stays loaded"
    )
