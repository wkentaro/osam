from typing import Optional

import fastapi
import numpy as np
import pydantic

from samuel import _json
from samuel import _models
from samuel import _types

app: fastapi.FastAPI = fastapi.FastAPI()

model: Optional[_models.ModelBase] = None


@app.get("/")
def index():
    return fastapi.Response("Samuel is running")


class GenerateRequest(pydantic.BaseModel):
    model: str
    image: str
    prompt: _types.Prompt


class GenerateResponse(pydantic.BaseModel):
    mask: str


@app.post("/api/generate")
def generate(request: GenerateRequest) -> GenerateResponse:
    global model

    if model is None or model.name != request.model:
        for model_cls in _models.MODELS:
            if model_cls.name == request.model:
                model = model_cls()
                break
        else:
            raise fastapi.HTTPException(404, "Model not found")

    image: np.ndarray = _json.image_b64data_to_ndarray(request.image)
    image_embedding: _types.ImageEmbedding = model.encode_image(image=image)
    mask: np.ndarray = model.generate_mask(
        image_embedding=image_embedding, prompt=request.prompt
    )

    return GenerateResponse(mask=_json.image_ndarray_to_b64data(mask))
