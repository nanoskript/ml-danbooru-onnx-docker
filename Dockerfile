FROM python:3.10-slim-buster AS base

RUN pip install --no-cache-dir pdm
ADD ./pyproject.toml ./pdm.lock ./
ADD ./common.py ./

FROM base AS builder

ARG MODEL
ARG IMAGE_SIZE

ADD ./ML-Danbooru-webui ./ML-Danbooru-webui
RUN pdm sync -G generate-onnx && pdm cache clear

ADD ./patch ./patch
ADD ./vendor ./vendor
ADD ./generate-onnx.py ./
RUN pdm run generate-onnx.py

FROM base

COPY --from=builder ./vendor ./vendor
RUN pdm sync -G server && pdm cache clear

ADD ./server.py ./
CMD pdm run uvicorn --host 0.0.0.0 --port $PORT server:app

