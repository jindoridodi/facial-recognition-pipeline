# syntax=docker/dockerfile:1

FROM python:3.11-slim

ARG INSIGHTFACE_REF=master

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/opt/insightface

# build-essential and cmake are required to build InsightFace's Python package.
# libgl1 and libglib2.0-0 provide OpenCV's common runtime dependencies.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install build prerequisites first, then install InsightFace directly from
# https://github.com/deepinsight/insightface and use ONNX Runtime on the CPU.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "numpy<2" cython \
    && python -m pip install \
        "git+https://github.com/deepinsight/insightface.git@${INSIGHTFACE_REF}#subdirectory=python-package" \
        onnxruntime \
        opencv-python-headless \
        fastapi \
        uvicorn

COPY app.py index.html /app/

RUN mkdir -p "${INSIGHTFACE_HOME}"

WORKDIR /app

# Persist downloaded model packs by mounting a volume at /opt/insightface.
VOLUME ["/opt/insightface"]

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
