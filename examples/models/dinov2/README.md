# DINOv2 Image Classification — ExecuTorch CUDA Backend

Export and run [DINOv2](https://huggingface.co/facebook/dinov2-small-imagenet1k-1-layer) image classification on the CUDA backend with ExecuTorch.

The default model is `facebook/dinov2-small-imagenet1k-1-layer`, a lightweight 1-layer variant of DINOv2 ViT-S/14 with an ImageNet-1k classification head. You can specify a different variant via `--model-name`.

## Prerequisites

- CUDA toolkit with `nvcc` on PATH (e.g., `/usr/local/cuda/bin`)
- Conda environment with ExecuTorch and transformers installed
- GPU with bf16 support

```bash
pip install -r install_requirements.txt
```

## Export

```bash
python -m executorch.examples.models.dinov2.export_dinov2 \
    --backend cuda \
    --output-dir ./dinov2_exports
```

This produces:
- `model.pte` — ExecuTorch program
- `aoti_cuda_blob.ptd` — CUDA delegate data

### Export Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `cuda` | `cuda` or `cuda-windows` |
| `--model-name` | `facebook/dinov2-small-imagenet1k-1-layer` | HuggingFace model name |
| `--dtype` | `bf16` | `bf16` or `fp32` |
| `--output-dir` | `./dinov2_exports` | Output directory |
| `--img-size` | `224` | Input image size |
| `--random-weights` | off | Use random weights for pipeline testing |

## Build

From the ExecuTorch root:

```bash
make dinov2-cuda
```

The binary is placed at `cmake-out/examples/models/dinov2/dinov2_runner`.

## Run

```bash
cmake-out/examples/models/dinov2/dinov2_runner \
    --model_path dinov2_exports/model.pte \
    --data_path dinov2_exports/aoti_cuda_blob.ptd \
    --image_path path/to/image.jpg
```

If `--image_path` is omitted, random input is used for testing.

### Runner Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `model.pte` | Path to `.pte` file |
| `--data_path` | (required) | Path to `.ptd` CUDA data file |
| `--image_path` | (none) | Input image (jpg/png/bmp) |
| `--img_size` | `224` | Input image size |
| `--top_k` | `5` | Number of top predictions to show |
| `--bf16` | `true` | Use bfloat16 input (should match export dtype) |

### Example Output

```
Output shape: (1, 1000)

Top-5 predictions:
  Class 258 (Samoyed): 14
  Class 259 (Pomeranian): 13.3125
  Class 261 (keeshond): 11.8125
  Class 257: 10.6875
  Class 152: 9.6875
```

## Troubleshooting

If you get `GLIBCXX_3.4.29` errors at runtime, add your conda lib to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```
