import typing
from typing import Final

import numpy as np
import numpy.typing as npt
import pydantic

Box = tuple[float, float, float, float]
PointLabel = typing.Literal[0, 1]
PromptKind = typing.Literal["points", "box", "texts", "example_boxes"]

# The models that find every match default these to 0.5, 0.1 and 100.
THRESHOLDS: Final = frozenset({"iou_threshold", "score_threshold", "max_annotations"})


class Prompt(pydantic.BaseModel):
    # One object: points with at most one box.
    points: list[tuple[float, float]] | None = pydantic.Field(
        default=None, description="x, y in pixels"
    )
    point_labels: list[PointLabel] | None = pydantic.Field(
        default=None, description="1 for foreground, 0 for background"
    )
    box: Box | None = pydantic.Field(
        default=None, description="xmin, ymin, xmax, ymax in pixels"
    )
    # Every match: texts, example boxes, or both.
    texts: list[str] | None = None
    example_boxes: list[Box] | None = pydantic.Field(
        default=None, description="xmin, ymin, xmax, ymax in pixels"
    )
    example_box_labels: list[PointLabel] | None = pydantic.Field(
        default=None,
        description="1 for a positive example, 0 for a negative one; all 1 if unset",
    )
    iou_threshold: float | None = None
    score_threshold: float | None = None
    max_annotations: int | None = pydantic.Field(default=None, ge=0)

    @property
    def kinds(self) -> frozenset[PromptKind]:
        # The kinds given, not empty; none asks for the model's default prompt.
        return frozenset(
            kind for kind in typing.get_args(PromptKind) if getattr(self, kind)
        )

    @pydantic.model_validator(mode="after")
    def _check_prompt(self) -> "Prompt":
        if (self.points is None) != (self.point_labels is None) or len(
            self.points or []
        ) != len(self.point_labels or []):
            raise ValueError("points and point_labels must have the same length")
        if self.example_box_labels is not None and len(self.example_box_labels) != len(
            self.example_boxes or []
        ):
            raise ValueError("example_box_labels must match example_boxes")
        one_object: bool = self.points is not None or self.box is not None
        if one_object and (self.texts is not None or self.example_boxes is not None):
            raise ValueError(
                "points and box segment one object, and texts and example_boxes "
                "find every match; give one or the other"
            )
        return self


def point_inputs(
    *, prompt: Prompt
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # The points and box as SAM-style decoders take them: labels 0 and 1 for
    # points, then 2 and 3 for the box's left-top and right-bottom corners.
    points: list = [*(prompt.points or [])]
    labels: list[int] = [*(prompt.point_labels or [])]
    if prompt.box is not None:
        points += [prompt.box[:2], prompt.box[2:]]
        labels += [2, 3]
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.float32)
