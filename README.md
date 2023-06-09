# ml-danbooru-onnx-docker

[Docker Hub](https://hub.docker.com/r/nanoskript/ml-danbooru)
| [API documentation](https://ml-danbooru.nanoskript.dev/docs)
| [Web tool](https://nanoskript.dev/tools/ml-danbooru/)

ONNX model and Docker service for <https://github.com/7eu7d7/ML-Danbooru>.

Exporting ML-Danbooru's models from PyTorch to ONNX significantly improves
inference times when running on CPU.

## Docker installation

```
docker run --publish $PORT:$PORT --env PORT=$PORT --detach nanoskript/ml-danbooru
```

## Configuration options

All configuration options are optional.

### At build time

- `MODEL` - The checkpoint model to be converted into an ONNX model.
  Currently, only `ml_caformer_m36_fp16_dec-5-97527.ckpt` is supported.
- `IMAGE_SIZE` - The resolution to scale input images to before processing them.
  Must be a multiple of `32`. The default is `256`.

Build time arguments can be provided with the `--build-arg` flag:

```
docker run --build-arg IMAGE_SIZE=256 ...
```

### At runtime

- `DEFAULT_THRESHOLD` - The default confidence threshold for which results are filtered
  by. The default is `0.5`.
- `MINIMUM_THRESHOLD` - The minimum confidence threshold for filtering that is allowed.
  Requests with a threshold lower than this will be rejected.

Runtime arguments can be provided with the `--env` flag:

```
docker run --env DEFAULT_THRESHOLD=0.5 ... 
```

## Generating a ONNX model manually

### System requirements

- Python 3.10 with the `pdm` package manager

### Steps

1. Clone this repository:

   ```
   git clone --recurse-submodules https://github.com/nanoskript/ml-danbooru-onnx-docker.git
   ```

2. Install dependencies:

   ```
   pdm sync -G generate-onnx
   ```

3. Run `generate-onnx.py`:

   ```
   pdm run generate-onnx.py
   ```

   A `model.onnx` file will be created in the `vendor` folder.
