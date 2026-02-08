# Summary

This example demonstrates how to export and run Google's [Gemma 3](https://huggingface.co/google/gemma-3-4b-it) vision-language multimodal model locally on ExecuTorch with CUDA backend support.

# Exporting the model
To export the model, we use [Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch), a repo that enables exporting models straight from the source - from HuggingFace's Transformers repo.

## Setting up Optimum ExecuTorch
Install through pip package:
```
pip install optimum-executorch
```

Or install from source:
```
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
python install_dev.py
```

## CUDA Support
This guide focuses on CUDA backend support for Gemma3, which provides accelerated performance on NVIDIA GPUs.

### Exporting with CUDA
```bash
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "cuda" \
  --dtype bfloat16 \
  --device cuda \
  --output_dir="path/to/output/dir"
```

This will generate:
- `model.pte` - The exported model
- `aoti_cuda_blob.ptd` - The CUDA kernel blob required for runtime

### Exporting with INT4 Quantization (Tile Packed)
For improved performance and reduced memory footprint, you can export Gemma3 with INT4 weight quantization using tile-packed format:

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
  --output_dir="path/to/output/dir"
```

This will generate the same files (`model.pte` and `aoti_cuda_blob.ptd`) in the `int4` directory.

See the "Building the Gemma3 runner" section below for instructions on building with CUDA support, and the "Running the model" section for runtime instructions.

# Running the model
To run the model, we will use the Gemma3 runner, which utilizes ExecuTorch's MultiModal runner API.
The Gemma3 runner will do the following:

- **Image Input**: Load image files (PNG, JPG, etc.) and format them as input tensors for the model
- **Text Input**: Process text prompts using the tokenizer
- **Feed the formatted inputs** to the multimodal runner for inference

## Obtaining the tokenizer
You can download the `tokenizer.json` file from [Gemma 3's HuggingFace repo](https://huggingface.co/unsloth/gemma-3-1b-it):
```bash
curl -L https://huggingface.co/unsloth/gemma-3-1b-it/resolve/main/tokenizer.json -o tokenizer.json
```

## Building the Gemma3 runner

### Prerequisites
Ensure you have a CUDA-capable GPU and CUDA toolkit installed on your system.

### Building for CUDA
```bash
# Build the Gemma3 runner with CUDA enabled
make gemma3-cuda

# Build the Gemma3 runner with CPU enabled
make gemma3-cpu
```

## Running the model
You need to provide the following files to run Gemma3:
- `model.pte` - The exported model file
- `aoti_cuda_blob.ptd` - The CUDA kernel blob
- `tokenizer.json` - The tokenizer file
- An image file (PNG, JPG, etc.)

### Example usage
```bash
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path path/to/model.pte \
  --data_path path/to/aoti_cuda_blob.ptd \
  --tokenizer_path path/to/tokenizer.json \
  --image_path docs/source/_static/img/et-logo.png \ # here we use the ExecuTorch logo as an example
  --temperature 0
```

# Example output
```
Okay, let's break down what's in the image!

It appears to be a stylized graphic combining:

*   **A Microchip:** The core shape is a representation of a microchip (the integrated circuit).
*   **An "On" Symbol:**  There's an "On" symbol (often represented as a circle with a vertical line) incorporated into the microchip design.
*   **Color Scheme:** The microchip is colored in gray, and
PyTorchObserver {"prompt_tokens":271,"generated_tokens":99,"model_load_start_ms":0,"model_load_end_ms":0,"inference_start_ms":1761118126790,"inference_end_ms":1761118128385,"prompt_eval_end_ms":1761118127175,"first_token_ms":1761118127175,"aggregate_sampling_time_ms":86,"SCALING_FACTOR_UNITS_PER_SECOND":1000}
```

# Running with Python Bindings

In addition to the C++ runner, you can also run Gemma 3 inference using ExecuTorch's Python bindings. This provides a more accessible interface for prototyping and integration with Python-based workflows.

## Prerequisites

Ensure you have ExecuTorch installed with Python bindings:
```bash
# In the executorch root directory
./install_executorch.sh
```

## Python Example

A complete Python binding example is provided in `pybinding_run.py`:

```bash
python examples/models/gemma3/pybinding_run.py \
  --model_path /path/to/model.pte \
  --tokenizer_path /path/to/tokenizer.json \
  --image_path /path/to/image.png \
  --prompt "What is in this image?"
```

### Key Implementation Details

The Python script demonstrates several important concepts:

1. **Loading Required Operators**: For quantized models, you must import the operator libraries before creating the runner:
   ```python
   import executorch.kernels.quantized  # For quantized ops
   from executorch.extension.llm.custom_ops import custom_ops  # For custom_sdpa
   ```

2. **Image Preprocessing**: Images must be preprocessed to match the model's expected input format:
   ```python
   from PIL import Image
   import torch
   import numpy as np

   def load_image(image_path: str, target_size: int = 896) -> torch.Tensor:
       pil_image = Image.open(image_path).convert("RGB")
       pil_image = pil_image.resize((target_size, target_size))
       # Convert HWC -> CHW, uint8 -> float32 [0, 1]
       image_tensor = (
           torch.from_numpy(np.array(pil_image))
           .permute(2, 0, 1)
           .contiguous()
           .float()
           / 255.0
       )
       return image_tensor
   ```

3. **Chat Template Format**: Gemma 3 uses a specific chat template for multimodal inputs:
   ```python
   from executorch.extension.llm.runner import make_text_input, make_image_input

   inputs = [
       make_text_input("<start_of_turn>user\n<start_of_image>"),
       make_image_input(image_tensor),
       make_text_input(f"{prompt}<end_of_turn>\n<start_of_turn>model\n"),
   ]
   ```

4. **Generation Configuration**:
   ```python
   from executorch.extension.llm.runner import GenerationConfig

   config = GenerationConfig(
       max_new_tokens=100,
       temperature=0.0,  # 0.0 for greedy decoding
       echo=False,
   )
   ```

5. **Running Inference with Callbacks**:
   ```python
   from executorch.extension.llm.runner import MultimodalRunner

   runner = MultimodalRunner(model_path, tokenizer_path)

   def token_callback(token: str):
       # Tokens are already printed by the C++ runner
       pass

   def stats_callback(stats):
       print(f"Prompt tokens: {stats.num_prompt_tokens}")
       print(f"Generated tokens: {stats.num_generated_tokens}")

   runner.generate(inputs, config, token_callback, stats_callback)
   ```

For more details on the Python API, see the [LLM Runner Framework documentation](../../../extension/llm/runner/README.md).

