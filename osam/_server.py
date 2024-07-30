import logging
import os

import fastapi

from . import apis
from . import types

app: fastapi.FastAPI = fastapi.FastAPI()


@app.get("/")
async def index():
    return fastapi.Response("osam is running")


@app.post("/api/generate")
async def generate(request: types.GenerateRequest) -> types.GenerateResponse:
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
