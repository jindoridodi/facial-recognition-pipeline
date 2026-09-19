### To Build and Run
docker build -f docker/Dockerfile -t insightface .
docker run --rm -p 8000:8000 -v insightface-models:/opt/insightface insightface

Open http://localhost:8000 and click **Start camera**, then **Detect faces**. The first detection downloads the InsightFace model pack into the mounted volume.

### To persist downloaded models
docker run --rm \
  -p 8000:8000 \
  -v insightface-models:/opt/insightface \
  insightface
