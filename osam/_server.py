import logging
import os

import fastapi
import pydantic
from pydantic.json_schema import SkipJsonSchema

from . import apis
from . import types

app: fastapi.FastAPI = fastapi.FastAPI()


# Image embeddings stay inside the server: clients send the image each time.
class GenerateRequest(types.GenerateRequest):
    image_embedding: SkipJsonSchema[None] = None


class GenerateResponse(pydantic.BaseModel):
    model: str
    annotations: list[types.Annotation]


@app.get("/")
async def index():
    return fastapi.Response("osam is running")


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> types.GenerateResponse:
    try:
        return apis.generate(request=request)
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))


# https://github.com/tiangolo/fastapi/issues/1508#issuecomment-638365277
@app.on_event("startup")
async def startup_event():
    uvicorn_logger = logging.getLogger("uvicorn")
    handler = logging.FileHandler(os.path.expanduser("~/.cache/osam/osam.log"))
    handler.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt))
    uvicorn_logger.addHandler(handler)
