import logging
import os
from typing import Optional

import fastapi
import numpy as np

from . import _json
from . import apis
from . import types

app: fastapi.FastAPI = fastapi.FastAPI()


@app.get("/")
async def index():
    return fastapi.Response("osam is running")


@app.post("/api/generate")
async def generate(request: types.GenerateRequest) -> types.GenerateResponse:
    try:
        image_array: Optional[np.ndarray] = None
        if request.image is not None:
            try:
                # Convert Base64 string to numpy array *after* validation
                image_array = _json.image_b64data_to_ndarray(b64data=request.image)
            except Exception as e:
                raise fastapi.HTTPException(
                    status_code=400, detail=f"Invalid Base64 image data: {e}"
                )

        # Create a copy of the request, replacing the string image with the numpy array
        backend_request = request.model_copy(
            update={"image": image_array}
        )

        # Pass the modified request to the backend
        return apis.generate(request=backend_request)
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
