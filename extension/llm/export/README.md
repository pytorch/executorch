# LLM Export API

This directory contains the unified API for exporting Large Language Models (LLMs) to ExecuTorch. The `export_llm` module provides a streamlined interface to convert various LLM architectures to optimized `.pte` files for on-device inference.

## Overview

The LLM export process transforms a model from its original format to an optimized representation suitable for mobile and edge devices. This involves several key steps:

1. **Model Instantiation**: Load the model architecture and weights from sources like Hugging Face
2. **Source Transformations**: Apply model-specific optimizations and quantization
3. **IR Export**: Convert to intermediate representations (EXIR, Edge dialect)
4. **Graph Transformations**: Apply backend-specific optimizations and PT2E quantization  
5. **Backend Delegation**: Partition operations to hardware-specific backends (XNNPACK, CoreML, QNN, etc.)
6. **Serialization**: Export to final ExecuTorch `.pte` format

## Supported Models

- **Llama**: Llama 2, Llama 3, Llama 3.1, Llama 3.2 (1B, 3B, 8B variants)
- **Qwen**: Qwen 2.5, Qwen 3 (0.6B, 1.7B, 4B variants)  
- **Phi**: Phi-3-Mini, Phi-4-Mini
- **Stories**: Stories110M (educational model)
- **SmolLM**: SmolLM2

## Usage

The export API supports two configuration approaches:

### Option 1: Hydra CLI Arguments

Use structured configuration arguments directly on the command line:

```bash
python -m extension.llm.export.export_llm \
    base.model_class=llama3 \
    model.use_sdpa_with_kv_cache=True \
    model.use_kv_cache=True \
    export.max_seq_length=128 \
    debug.verbose=True \
    backend.xnnpack.enabled=True \
    backend.xnnpack.extended_ops=True \
    quantization.qmode=8da4w
```

### Option 2: Configuration File

Create a YAML configuration file and reference it:

```bash
python -m extension.llm.export.export_llm --config my_config.yaml
```

Example `my_config.yaml`:
```yaml
base:
  model_class: llama3
  tokenizer_path: /path/to/tokenizer.json

model:
  use_kv_cache: true
  use_sdpa_with_kv_cache: true
  enable_dynamic_shape: true

export:
  max_seq_length: 512
  output_dir: ./exported_models
  output_name: llama3_optimized.pte

quantization:
  qmode: 8da4w
  group_size: 32

backend:
  xnnpack:
    enabled: true
    extended_ops: true

debug:
  verbose: true
```

**Important**: You cannot mix both approaches. Use either CLI arguments OR a config file, not both.

## Example Commands

### Export Qwen3 0.6B with XNNPACK backend and quantization
```bash
python -m extension.llm.export.export_llm \
    base.model_class=qwen3_0_6b \
    base.params=examples/models/qwen3/0_6b_config.json \
    base.metadata='{"get_bos_id": 151644, "get_eos_ids":[151645]}' \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    model.dtype_override=FP32 \
    export.max_seq_length=512 \
    export.output_name=qwen3_0_6b.pte \
    quantization.qmode=8da4w \
    backend.xnnpack.enabled=true \
    backend.xnnpack.extended_ops=true \
    debug.verbose=true
```

### Export Phi-4-Mini with custom checkpoint
```bash
python -m extension.llm.export.export_llm \
    base.model_class=phi_4_mini \
    base.checkpoint=/path/to/phi4_checkpoint.pth \
    base.params=examples/models/phi-4-mini/config.json \
    base.metadata='{"get_bos_id":151643, "get_eos_ids":[151643]}' \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    export.max_seq_length=256 \
    export.output_name=phi4_mini.pte \
    backend.xnnpack.enabled=true \
    debug.verbose=true
```

### Export with CoreML backend (iOS optimization)
```bash
python -m extension.llm.export.export_llm \
    base.model_class=llama3 \
    model.use_kv_cache=true \
    export.max_seq_length=128 \
    backend.coreml.enabled=true \
    backend.coreml.compute_units=ALL \
    quantization.pt2e_quantize=coreml_c4w \
    debug.verbose=true
```

## Configuration Options

For a complete reference of all available configuration options, see the [LlmConfig class definition](../../../examples/models/llama/config/llm_config.py) which documents all supported parameters for base, model, export, quantization, backend, and debug configurations.

## Further Reading

- [Llama Examples](../../../examples/models/llama/README.md) - Comprehensive Llama export guide
- [LLM Runner](../runner/) - Running exported models
- [ExecuTorch Documentation](https://pytorch.org/executorch/) - Framework overview