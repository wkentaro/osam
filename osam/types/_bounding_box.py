import pydantic


class BoundingBox(pydantic.BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @pydantic.model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.xmin > self.xmax or self.ymin > self.ymax:
            raise ValueError(f"bounding box {self} is inverted")
        return self
