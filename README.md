## Build and run

```sh
docker build -f docker/Dockerfile -t insightface .
docker run --rm -p 8000:8000 insightface
```

Open [http://localhost:8000](http://localhost:8000).

## Live updates

```sh
docker run --rm -p 8000:8000 -v "$PWD":/app insightface uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
