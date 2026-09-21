## Build and run

```sh
docker build -f docker/Dockerfile -t insightface .
docker run --rm -p 8000:8000 \
  -v insightface-models:/opt/insightface \
  -v facial-embeddings:/data \
  insightface
```

Open [http://localhost:8000](http://localhost:8000).

The webcam pipeline currently detects faces, returns five landmarks per face
(eyes, nose, and mouth corners), estimates an ArcFace similarity transform, and
uses an affine warp to create the recognition model's normalized 112×112 input.
The aligned OpenCV image is converted from BGR to RGB, normalized with
`(pixel - 127.5) / 127.5`, and rearranged from HWC to a contiguous float32 NCHW
tensor with shape `[1, 3, 112, 112]` for the future recognition stage.
The API response from `POST /api/detect` includes `landmarks`, an `aligned`
JPEG data URL, and an `identity` for each face. InsightFace also produces a
512-dimensional, L2-normalized ArcFace embedding as a private NumPy `float32`
array. Detection responses expose only embedding metadata, never vector values.
An unmatched face has an `identity` of `null`; a match includes the saved name,
embedding ID, and cosine similarity.

The browser assigns session-local track IDs using face position, motion, and
recognized identity. Aligned previews remain in track order even if the model's
per-frame detection order changes. Tracks survive five missed frames and reset
when the camera restarts; the original per-frame index is retained privately in
the browser so enrollment still selects the correct face from its exact image.

## Embedding enrollment

Use the name field beneath any detected face in the browser to save it. The UI
enrolls the selected face from the exact frame shown in that face preview and
then displays recognized names beneath their camera boxes. Saving the same name
again adds another sample, which can improve recognition across different
angles and lighting.

Embeddings can also be persisted through the enrollment endpoint. `subject_id`
is the person's display name; it is trimmed and limited to 128 characters. If
an image contains multiple faces, pass the zero-based `face_index` from the
detection order:

```sh
curl -X POST http://localhost:8000/api/enroll \
  -H 'Content-Type: application/json' \
  -d '{"subject_id":"Ada Lovelace","face_index":0,"image":"data:image/jpeg;base64,..."}'
```

For backward compatibility, `face_index` may be omitted when the image contains
exactly one face.

The response contains an `embedding_id` UUID plus non-vector metadata. Delete a
stored sample with:

```sh
curl -X DELETE http://localhost:8000/api/embeddings/EMBEDDING_ID
```

Use the **Clear saved faces** button in the browser, or delete every saved
embedding through the API:

```sh
curl -X DELETE http://localhost:8000/api/embeddings
```

Embeddings are stored as little-endian float32 BLOBs in SQLite. Local runs use
`data/embeddings.sqlite3`; set `EMBEDDING_DB_PATH` to override it. Docker uses
`/data/embeddings.sqlite3`, so mount a volume at `/data` to persist enrollment
records across containers. Records include the `buffalo_l` model name and an
embedding schema version because vectors from different recognition models are
not directly comparable.

Live recognition uses cosine similarity and accepts the closest compatible
saved embedding at a default threshold of `0.4`. Set `FACE_MATCH_THRESHOLD` to
a value from `-1.0` to `1.0` to tune it; raising the threshold reduces false
matches but may leave more faces unidentified.

This prototype has no API authentication or database encryption. Facial
embeddings are biometric data; expose these endpoints only in a trusted local
environment.

## Project structure

```text
app.py                         FastAPI application assembly
assets/face-tracker.js         Browser-session face association and ordering
facial_pipeline/
  config.py                    Model and path configuration
  schemas.py                   API request models
  image_ops.py                 Decode, align, normalize, and reshape operations
  embeddings.py                Embedding validation and metadata
  embedding_store.py           SQLite embedding persistence
  matching.py                  Cosine matching and public identity metadata
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
Reuse that named volume on later `docker run` commands to retain names and
facial data after replacing a container.
