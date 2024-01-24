import fastapi

app = fastapi.FastAPI()


@app.get("/")
def index():
    return fastapi.Response("Samuel is running")
