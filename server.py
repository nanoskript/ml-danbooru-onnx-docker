import io
import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime
import torch
from PIL import Image
from pydantic.main import BaseModel
from fastapi import FastAPI, File, Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, JSONResponse

from common import prepare_image

# Server options.
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD") or "0.5")
MINIMUM_THRESHOLD = float(os.environ.get("MINIMUM_THRESHOLD") or "0.2")

# Load JSON files.
with open(Path("vendor") / "build.json") as f:
    options = json.load(f)
with open(Path("vendor") / "class_map.json") as f:
    class_map = json.load(f)

# Start inference session.
session = onnxruntime.InferenceSession(Path("vendor") / "model.onnx")

# Server.
app = FastAPI(title="ml-danbooru-onnx-docker")
app.add_middleware(CORSMiddleware, allow_origins=["*"])


class Prediction(BaseModel):
    tag: str
    score: float


def infer(image: Image, threshold: float):
    tensor_input = prepare_image(image, options["IMAGE_SIZE"])
    inputs = {session.get_inputs()[0].name: tensor_input.numpy()}
    outputs = session.run(None, inputs)

    tensor = torch.tensor(np.array(outputs))
    output = torch.sigmoid(tensor).view(-1)
    predictions = torch.where(output > threshold)[0].numpy()

    results = [
        Prediction(
            tag=class_map[str(p)],
            score=float(output[p])
        )
        for p in predictions
    ]

    results.sort(reverse=True, key=lambda p: p.score)
    return results


@app.get("/", include_in_schema=False)
async def route_index():
    return RedirectResponse("/docs")


@app.get("/configuration", summary="Get the build time configuration.")
async def route_configuration():
    return JSONResponse(options)


@app.post("/ml-danbooru", summary="Extract Danbooru tags from an image.")
async def route_ml_danbooru(
    image: bytes = File(),
    threshold: Annotated[float, Form(ge=MINIMUM_THRESHOLD, le=1.0)] = DEFAULT_THRESHOLD,
) -> list[Prediction]:
    return infer(Image.open(io.BytesIO(image)), threshold)
