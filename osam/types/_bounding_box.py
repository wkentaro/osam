import pydantic


class BoundingBox(pydantic.BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
