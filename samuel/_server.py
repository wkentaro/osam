import fastapi

from samuel import apis
from samuel import types

app: fastapi.FastAPI = fastapi.FastAPI()


@app.get("/")
def index():
    return fastapi.Response("Samuel is running")


@app.post("/api/generate_mask")
def generate(request: types.GenerateMaskRequest) -> types.GenerateMaskResponse:
    try:
        return apis.generate_mask(request=request)
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
