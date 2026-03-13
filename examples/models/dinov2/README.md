# DINOv2 Image Classification — ExecuTorch CUDA Backend

This example exports the [DINOv2](https://huggingface.co/facebook/dinov2-small-imagenet1k-1-layer) image classification model to ExecuTorch and runs inference on the CUDA backend.

## Prerequisites

- CUDA toolkit with `nvcc` on PATH (e.g., `/usr/local/cuda/bin`)
- Conda environment with ExecuTorch and transformers installed
- GPU with bf16 support

```bash
conda activate dinov2
pip install -r install_requirements.txt
```

## Export

Export the model to ExecuTorch `.pte` format with CUDA backend:

```bash
# Ensure nvcc is on PATH
export PATH="/usr/local/cuda/bin:$PATH"

# Export with pretrained weights (default: bf16)
python examples/models/dinov2/export_dinov2.py \
    --backend cuda \
    --output-dir ./dinov2_exports

# Export with random weights (for pipeline testing)
python examples/models/dinov2/export_dinov2.py \
    --backend cuda \
    --output-dir ./dinov2_exports \
    --random-weights
```

This produces:
- `model.pte` — ExecuTorch program (~1.7 MB)
- `aoti_cuda_blob.ptd` — CUDA delegate data (~44 MB)

### Export Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `facebook/dinov2-small-imagenet1k-1-layer` | HuggingFace model name |
| `--backend` | `cuda` | Backend: `cuda`, `cuda-windows`, `xnnpack`, `portable` |
| `--dtype` | `bf16` | Data type (`bf16` required for CUDA Triton SDPA) |
| `--output-dir` | `./dinov2_exports` | Output directory |
| `--img-size` | `224` | Input image size |
| `--random-weights` | off | Use random weights for testing |

## Python Inference

```python
import torch
from executorch.extension.pybindings._portable_lib import (
    _load_for_executorch_from_buffer,
    Verification,
)

with open("dinov2_exports/model.pte", "rb") as f:
    pte_data = f.read()
with open("dinov2_exports/aoti_cuda_blob.ptd", "rb") as f:
    ptd_data = f.read()

module = _load_for_executorch_from_buffer(
    pte_data,
    data_map_buffer=ptd_data,
    program_verification=Verification.Minimal,
)

input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
outputs = module.forward([input_tensor])
logits = outputs[0]  # shape: (1, 1000)

# Top-5 predictions
topk = torch.topk(logits.float(), 5)
for val, idx in zip(topk.values, topk.indices):
    print(f"Class {idx.item()}: {val.item():.4f}")
```

**Note:** On some systems you may need to set `LD_LIBRARY_PATH` to include a libstdc++ with GLIBCXX_3.4.30+ (e.g., from the conda env):

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## C++ Inference (Runner)

### Build

Requires ExecuTorch built from source with CUDA support:

```bash
# From executorch root, build the runner
cmake --workflow --preset dinov2-cuda
```

### Run

```bash
./cmake-out/examples/models/dinov2/dinov2_runner \
    --model_path dinov2_exports/model.pte \
    --data_path dinov2_exports/aoti_cuda_blob.ptd \
    --input_path image.raw
```

The `--input_path` expects a raw float32 binary file of shape `(1, 3, 224, 224)`. If omitted, random input is used for testing.

## Model Details

- **Architecture:** DINOv2 ViT-S/14 with ImageNet-1k classification head
- **Input:** `(1, 3, 224, 224)` image tensor (bf16)
- **Output:** `(1, 1000)` classification logits (bf16)
- **Parameters:** ~22M

## Accuracy

With bf16 precision and pretrained weights, ExecuTorch CUDA output is consistent with eager PyTorch:

- **Top-1 class match:** Yes
- **Cosine similarity (logits):** ~0.93
