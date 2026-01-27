# Summary

This example demonstrates how to export and run Google's Gemma 3 models on ExecuTorch:
- [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it) - Vision-language multimodal model (CUDA/CPU)
- [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it) - Text-only instruction-tuned model (CPU)

# Prerequisites

## Setting up Optimum ExecuTorch
To export the models, we use [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch), which enables exporting models from HuggingFace's Transformers.

Install through pip:
```bash
pip install optimum-executorch
```

Or install from source:
```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
python install_dev.py
```

## Obtaining the Tokenizer
Both Gemma 3 models share the same tokenizer. Download `tokenizer.json` from HuggingFace:
```bash
mkdir -p gemma-3
curl -L https://huggingface.co/google/gemma-3-1b-it/resolve/main/tokenizer.json -o gemma-3/tokenizer.json
```

---

# Gemma 3 1B Text-Only Model (CPU)

This section covers running the lightweight Gemma 3 1B instruction-tuned model for text-only inference on CPU.

## Exporting Gemma 3 1B

```bash
optimum-cli export executorch \
  --model "google/gemma-3-1b-it" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --output_dir="gemma-3/gemma-3-1b-it"
```

This will generate:
- `model.pte` - The exported model

## Building the Text Runner

```bash
make gemma3-text-cpu
```

## Running the Text Model

```bash
./cmake-out/examples/models/gemma3/gemma3_text_runner \
    --model_path=gemma-3/gemma-3-1b-it/model.pte \
    --tokenizer_path=gemma-3/tokenizer.json \
    --prompt="What is the capital of France?" \
    --max_new_tokens=100
```

### Available Options
| Flag | Description | Default |
|------|-------------|---------|
| `--model_path` | Path to the exported model.pte | `model.pte` |
| `--tokenizer_path` | Path to tokenizer.json | `tokenizer.json` |
| `--prompt` | Text prompt for generation | `Hello, world!` |
| `--temperature` | Sampling temperature (0 = greedy) | `0.0` |
| `--max_new_tokens` | Maximum tokens to generate | `100` |
| `--cpu_threads` | Number of CPU threads (-1 = auto) | `-1` |
| `--warmup` | Run warmup before generation | `false` |

### Example Output
```
The capital of France is **Paris**.
PyTorchObserver {"prompt_tokens":15,"generated_tokens":12,...}
```

---

# Gemma 3 4B Multimodal Model (CUDA)

This section covers running the Gemma 3 4B vision-language multimodal model with CUDA backend support.

## Exporting Gemma 3 4B

### Standard Export
```bash
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "cuda" \
  --dtype bfloat16 \
  --device cuda \
  --output_dir="gemma-3/gemma-3-4b-it"
```

This will generate:
- `model.pte` - The exported model
- `aoti_cuda_blob.ptd` - The CUDA kernel blob required for runtime

### Export with INT4 Quantization (Tile Packed)
For improved performance and reduced memory footprint:

```bash
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "cuda" \
  --dtype bfloat16 \
  --device cuda \
  --qlinear 4w \
  --qlinear_encoder 4w \
  --qlinear_packing_format tile_packed_to_4d \
  --qlinear_encoder_packing_format tile_packed_to_4d \
  --output_dir="gemma-3/gemma-3-4b-it-int4"
```

## Building the Multimodal Runner

### Prerequisites
Ensure you have a CUDA-capable GPU and CUDA toolkit installed.

### Build Commands
```bash
# Build with CUDA backend
make gemma3-cuda

# Build with CPU backend
make gemma3-cpu
```

## Running the Multimodal Model

The multimodal runner processes both image and text inputs:

```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path=gemma-3/gemma-3-4b-it/model.pte \
  --data_path=gemma-3/gemma-3-4b-it/aoti_cuda_blob.ptd \
  --tokenizer_path=gemma-3/tokenizer.json \
  --image_path=docs/source/_static/img/et-logo.png \
  --temperature=0
```

### Required Files
| File | Description |
|------|-------------|
| `model.pte` | The exported model file |
| `aoti_cuda_blob.ptd` | CUDA kernel blob (CUDA only) |
| `tokenizer.json` | Shared tokenizer |
| Image file | PNG, JPG, or other supported format |

### Example Output
```
Okay, let's break down what's in the image!

It appears to be a stylized graphic combining:

*   **A Microchip:** The core shape is a representation of a microchip (the integrated circuit).
*   **An "On" Symbol:**  There's an "On" symbol (often represented as a circle with a vertical line) incorporated into the microchip design.
*   **Color Scheme:** The microchip is colored in gray, and
PyTorchObserver {"prompt_tokens":271,"generated_tokens":99,...}
```
