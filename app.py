import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from facial_pipeline.config import PROJECT_ROOT
from facial_pipeline.routes import router as api_router


logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Facial Recognition Prototype")
app.mount("/assets", StaticFiles(directory=PROJECT_ROOT / "assets"), name="assets")
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")
