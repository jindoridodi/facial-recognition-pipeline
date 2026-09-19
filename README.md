## Build and run

```sh
docker build -f docker/Dockerfile -t insightface .
docker run --rm -p 8000:8000 insightface
```

Open [http://localhost:8000](http://localhost:8000).

The webcam pipeline currently detects faces, returns five landmarks per face
(eyes, nose, and mouth corners), and creates a normalized 112×112 aligned crop.
The API response from `POST /api/detect` includes `landmarks` and an `aligned`
JPEG data URL for each face. Recognition and identity matching are intentionally
not part of this stage.

## Live updates

```sh
docker run --rm -p 8000:8000 -v "$PWD":/app insightface uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
