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

The export API supports a Hydra-style CLI where you can you configure using yaml and also CLI args.

### Hydra CLI Arguments

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

### Configuration File

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

You can you also still provide additional overrides using the CLI args as well:

```bash
python -m extension.llm.export.export_llm
  --config my_config.yaml
  base.model_class="llama2"
  +export.max_context_length=1024
```

Note that if a config file is specified and you want to specify a CLI arg that is not in the config, you need to prepend with a `+`. You can read more about this in the Hydra [docs](https://hydra.cc/docs/advanced/override_grammar/basic/).


## Example Commands

Please refer to the docs for some of our example suported models ([Llama](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md), [Qwen3](https://github.com/pytorch/executorch/tree/main/examples/models/qwen3/README.md), [Phi-4-mini](https://github.com/pytorch/executorch/tree/main/examples/models/phi_4_mini/README.md)).

## Configuration Options

For a complete reference of all available configuration options, see the [LlmConfig class definition](config/llm_config.py) which documents all supported parameters for base, model, export, quantization, backend, and debug configurations.

## Further Reading

- [Llama Examples](../../../examples/models/llama/README.md) - Comprehensive Llama export guide
- [LLM Runner](../runner/) - Running exported models
- [ExecuTorch Documentation](https://pytorch.org/executorch/) - Framework overview
