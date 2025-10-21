# Exporting LLMs with HuggingFace's Optimum ExecuTorch

[Optimum ExecuTorch](https://github.com/huggingface/optimum-executorch) provides a streamlined way to export Hugging Face transformer models to ExecuTorch format. It offers seamless integration with the Hugging Face ecosystem, making it easy to export models directly from the Hugging Face Hub.

## Overview

Optimum ExecuTorch supports a much wider variety of model architectures compared to ExecuTorch's native `export_llm` API. While `export_llm` focuses on a limited set of highly optimized models (Llama, Qwen, Phi, and SmolLM) with advanced features like SpinQuant and attention sink, Optimum ExecuTorch can export diverse architectures including Gemma, Mistral, GPT-2, BERT, T5, Whisper, Voxtral, and many others.

### Use Optimum ExecuTorch when:
- You need to export models beyond the limited set supported by `export_llm`
- Exporting directly from Hugging Face Hub model IDs, including model variants such as finetunes
- You want a simpler interface with Hugging Face ecosystem integration

### Use export_llm when:
- Working with one of the highly optimized supported models (Llama, Qwen, Phi, SmolLM)
- You need advanced optimizations like SpinQuant or attention sink
- You need pt2e quantization for QNN/CoreML/Vulkan backends
- Working with Llama models requiring custom checkpoints

See [Exporting LLMs](export-llm.md) for details on using the native `export_llm` API.

## Prerequisites

### Installation

First, clone and install Optimum ExecuTorch from source:

```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install '.[dev]'
```

For access to the latest features and optimizations, install dependencies in dev mode:

```bash
python install_dev.py
```

This installs `executorch`, `torch`, `torchao`, `transformers`, and other dependencies from nightly builds or source.

## Supported Models

Optimum ExecuTorch supports a wide range of model architectures including decoder-only LLMs (Llama, Qwen, Gemma, Mistral, etc.), multimodal models, vision models, audio models (Whisper), encoder models (BERT, RoBERTa), and seq2seq models (T5).

For the complete list of supported models, see the [Optimum ExecuTorch documentation](https://github.com/huggingface/optimum-executorch#-supported-models).

## Export Methods

Optimum ExecuTorch offers two ways to export models:

### Method 1: CLI Export

The CLI is the simplest way to export models. It provides a single command to convert models from Hugging Face Hub to ExecuTorch format.

#### Basic Export

```bash
optimum-cli export executorch \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --output_dir="./smollm2_exported"
```

#### With Optimizations

Add custom SDPA, KV cache optimization, and quantization:

```bash
optimum-cli export executorch \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="./smollm2_exported"
```

#### Available CLI Arguments

Key arguments for LLM export include `--model`, `--task`, `--recipe` (backend), `--use_custom_sdpa`, `--use_custom_kv_cache`, `--qlinear` (linear quantization), `--qembedding` (embedding quantization), and `--max_seq_len`.

For the complete list of arguments, run:
```bash
optimum-cli export executorch --help
```

## Optimization Options

### Custom Operators

Optimum ExecuTorch includes custom SDPA (~3x speedup) and custom KV cache (~2.5x speedup) operators. Enable with `--use_custom_sdpa` and `--use_custom_kv_cache`.

### Quantization

Optimum ExecuTorch uses [TorchAO](https://github.com/pytorch/ao) for quantization. Common options:
- `--qlinear 8da4w`: int8 dynamic activation + int4 weight (recommended)
- `--qembedding 4w` or `--qembedding 8w`: int4/int8 embedding quantization

Example:
```bash
optimum-cli export executorch \
    --model "meta-llama/Llama-3.2-1B" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 4w \
    --output_dir="./llama32_1b"
```

### Backend Support

Supported backends: `xnnpack` (CPU), `coreml` (Apple GPU), `portable` (baseline), `cuda` (NVIDIA GPU). Specify with `--recipe`.

## Exporting Different Model Types

Optimum ExecuTorch supports various model architectures with different tasks:

- **Decoder-only LLMs**: Use `--task text-generation`
- **Multimodal LLMs**: Use `--task multimodal-text-to-text`
- **Seq2Seq models** (T5): Use `--task text2text-generation`
- **ASR models** (Whisper): Use `--task automatic-speech-recognition`

For detailed examples of exporting each model type, see the [Optimum ExecuTorch export guide](https://github.com/huggingface/optimum-executorch/blob/main/optimum/exporters/executorch/README.md).

## Running Exported Models

### Verifying Output with Python

After exporting, you can verify the model output in Python before deploying to device using classes from `modeling.py`, such as the `ExecuTorchModelForCausalLM` class for LLMs:

```python
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# Load the exported model
model = ExecuTorchModelForCausalLM.from_pretrained("./smollm2_exported")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Generate text
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_seq_len=128,
)
print(generated_text)
```

### Running on Device

After verifying your model works correctly, deploy it to device:

- [Running with C++](run-with-c-plus-plus.md) - Run exported models using ExecuTorch's C++ runtime
- [Running on Android](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/android) - Deploy to Android devices
- [Running on iOS](https://github.com/meta-pytorch/executorch-examples/tree/main/llm/apple) - Deploy to iOS devices

## Performance

For performance benchmarks and on-device metrics, see the [Optimum ExecuTorch benchmarks](https://github.com/huggingface/optimum-executorch#-benchmarks-on-mobile-devices) and the [ExecuTorch Benchmark Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch).

## Additional Resources

- [Optimum ExecuTorch GitHub](https://github.com/huggingface/optimum-executorch) - Full documentation and examples
- [Supported Models](https://github.com/huggingface/optimum-executorch#-supported-models) - Complete model list
- [Export Guide](https://github.com/huggingface/optimum-executorch/blob/main/optimum/exporters/executorch/README.md) - Detailed export examples
- [TorchAO Quantization](https://github.com/pytorch/ao) - Quantization library documentation
