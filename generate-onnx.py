import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from common import prepare_image

sys.path.append("patch")
sys.path.append("ML-Danbooru-webui")

from mldanbooru.interface import Infer

# Model configuration.
DEFAULT_MODEL = "ml_caformer_m36_fp16_dec-5-97527.ckpt"
MODEL = os.environ.get("MODEL") or DEFAULT_MODEL
assert DEFAULT_MODEL in Infer.MODELS

# Image rescaling configuration.
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE") or "256")
assert IMAGE_SIZE % 32 == 0

infer = Infer()
infer.load_model(MODEL)
infer.model.eval()

# Write configuration.
with open(Path("vendor") / "build.json", "w") as f:
    json.dump({
        "MODEL": MODEL,
        "IMAGE_SIZE": IMAGE_SIZE,
    }, f)

# Write class map.
with open(Path("vendor") / "class_map.json", "w") as f:
    json.dump(infer.class_map, f)

# Generate model.
blank_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
tensor_input = prepare_image(blank_image, IMAGE_SIZE)
with open(Path("vendor") / "model.onnx", "wb") as f:
    torch.onnx.export(infer.model, tensor_input, f)
