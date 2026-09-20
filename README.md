## Build and run

```sh
docker build -f docker/Dockerfile -t insightface .
docker run --rm -p 8000:8000 insightface
```

Open [http://localhost:8000](http://localhost:8000).

The webcam pipeline currently detects faces, returns five landmarks per face
(eyes, nose, and mouth corners), estimates an ArcFace similarity transform, and
uses an affine warp to create the recognition model's normalized 112×112 input.
The aligned OpenCV image is converted from BGR to RGB, normalized with
`(pixel - 127.5) / 127.5`, and rearranged from HWC to a contiguous float32 NCHW
tensor with shape `[1, 3, 112, 112]` for the future recognition stage.
The API response from `POST /api/detect` includes `landmarks` and an `aligned`
JPEG data URL for each face. InsightFace also produces a 512-dimensional,
L2-normalized ArcFace embedding as a private NumPy `float32` array. Detection
responses expose only embedding metadata, never the vector values.

## Embedding enrollment

Persist an embedding only through the explicit enrollment endpoint. The image
must contain exactly one face, and `subject_id` is trimmed and limited to 128
characters.

```sh
curl -X POST http://localhost:8000/api/enroll \
  -H 'Content-Type: application/json' \
  -d '{"subject_id":"person-123","image":"data:image/jpeg;base64,..."}'
```

The response contains an `embedding_id` UUID plus non-vector metadata. Delete a
stored sample with:

```sh
curl -X DELETE http://localhost:8000/api/embeddings/EMBEDDING_ID
```

Embeddings are stored as little-endian float32 BLOBs in SQLite. Local runs use
`data/embeddings.sqlite3`; set `EMBEDDING_DB_PATH` to override it. Docker uses
`/data/embeddings.sqlite3`, so mount a volume at `/data` to persist enrollment
records across containers. Records include the `buffalo_l` model name and an
embedding schema version because vectors from different recognition models are
not directly comparable.

This prototype has no API authentication or database encryption. Facial
embeddings are biometric data; expose these endpoints only in a trusted local
environment. Identity matching and similarity thresholds are intentionally not
part of this stage.

## Project structure

```text
app.py                         FastAPI application assembly
facial_pipeline/
  config.py                    Model and path configuration
  schemas.py                   API request models
  image_ops.py                 Decode, align, normalize, and reshape operations
  embeddings.py                Embedding validation and metadata
  embedding_store.py           SQLite embedding persistence
  models.py                    Private pipeline result types
  service.py                   InsightFace loading and pipeline orchestration
  routes.py                    API endpoints under `/api`
```

## Live updates

```sh
docker run --rm -p 8000:8000 \
  -v "$PWD":/app \
  -v insightface-models:/opt/insightface \
  -v facial-embeddings:/data \
  insightface uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The `/app` bind mount applies source changes immediately, and the named
`insightface-models` volume keeps downloaded model packs between container
runs. The `facial-embeddings` volume keeps the SQLite enrollment database.
