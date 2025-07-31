# ExecuTorch Optimum Module

This module provides ExecuTorch-specific optimizations and integrations for transformer models. It focuses on runtime-specific features that are not available in the upstream transformers or optimum-executorch libraries.

## Overview

This streamlined module contains only ExecuTorch-specific components:

- Custom cache implementations optimized for ExecuTorch runtime
- Custom SDPA implementations for ExecuTorch operators
- XNNPACK backend integration and optimization passes
- ExecuTorch-specific utilities

For general model export functionality, use `optimum-executorch` which provides a comprehensive recipe system and CLI interface.

## Key Components

### Custom Cache Implementations

#### `ETCustomStaticCache` and `ETCustomHybridCache`
Custom KV cache implementations that inherit from Hugging Face's caches but use ExecuTorch's `CustomKVCache` and `CustomRingKVCache` for optimal runtime performance.

### Custom SDPA

#### `get_custom_sdpa_for_ring_kv_cache`
Custom Scaled Dot-Product Attention implementation optimized for ExecuTorch's ring buffer caches and sliding window attention.

### XNNPACK Integration

#### `export_to_executorch_with_xnnpack`
ExecuTorch-specific XNNPACK backend integration with custom optimization passes:
- `RemovePaddingIdxEmbeddingPass`: Removes padding_idx from embedding operations
- Memory planning and quantization optimizations
- Backend delegation analysis and debugging

### Utilities

- `save_config_to_constant_methods`: ExecuTorch-specific configuration utilities
- Model metadata extraction for runtime optimization

## Usage

For multimodal model export, use optimum-executorch:

```bash
# Export with optimum-executorch CLI
optimum-cli export executorch \
    --model google/gemma-3-4b-it \
    --task image-text-to-text \
    --recipe xnnpack \
    --use_custom_sdpa \
    --use_custom_kv_cache
```

```python
# Or via Python API
from optimum.executorch import ExecuTorchModelForCausalLM

model = ExecuTorchModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    task="image-text-to-text", 
    recipe="xnnpack",
    use_custom_sdpa=True,
    use_custom_kv_cache=True
)
```

For ExecuTorch-specific XNNPACK optimizations:

```python
from optimum.exporters.executorch.integrations import ImageTextToTextExportableModule
from executorch.extension.llm.optimum.xnnpack import export_to_executorch_with_xnnpack

# Load model using optimum-executorch
module = ImageTextToTextExportableModule(model, use_custom_kv_cache=True, use_custom_sdpa=True)

# Apply ExecuTorch-specific XNNPACK optimizations
executorch_program = export_to_executorch_with_xnnpack(module)
```

## Architecture

This module follows the recommended approach:
1. **General export functionality**: Use `optimum-executorch` 
2. **Multimodal support**: Enhanced `transformers.integrations.executorch`
3. **ExecuTorch-specific optimizations**: This module

This separation ensures:
- No code duplication between repositories
- Leverages mature optimum-executorch infrastructure  
- Focuses ExecuTorch module on runtime-specific optimizations
- Maintains unified user experience through optimum-executorch CLI/API

## Testing

Run tests with:
```bash
python -m pytest extension/llm/optimum/test/
```
