# ExecuTorch Optimum Module

This module provides integration utilities for exporting and optimizing transformer models for ExecuTorch runtime. It contains specialized wrapper classes and utilities to make pre-trained models from Hugging Face Transformers compatible with `torch.export` and ExecuTorch execution. A lot of code is forked from `optimum-executorch` and adopted from `transformers`. We put it in ExecuTorch so that we can fast iterate on the stack. Eventually we want to upstream changes to `transformers` and `optimum-executorch`.

## Overview

The optimum module bridges the gap between Hugging Face Transformers models and ExecuTorch by providing:

- Exportable wrapper modules for different model types
- Custom cache implementations for efficient inference
- Utilities for model configuration and optimization
- Integration with ExecuTorch's custom operators

## Key Components

### Exportable Modules

#### `TorchExportableModuleWithHybridCache`
A wrapper module that makes decoder-only language models exportable with `torch.export` using `HybridCache`. This is a forked version of [`TorchExportableModuleForDecoderOnlyLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py#L391) with some modifications to support `inputs_embeds`.

**Note**: This class should be upstreamed to transformers. We keep it here so that we can iterate quickly.

#### `TorchExportableModuleForImageTextLM`
A wrapper for text decoder model in a vision-language model. It is very similar to [`TorchExportableModuleForDecoderOnlyLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py#L30) but instead of taking `input_ids` this module takes `inputs_embeds`. This is because we want to be able to take both token embeddings and image embeddings as inputs.

**Note**: This class should be upstreamed to transformers. We keep it here so that we can iterate quickly.

#### `ImageEncoderExportableModule`
A wrapper for vision encoder models that projects vision features to language model space. Commonly implemented as `get_image_features()` in HuggingFace transformers. For example: [`Gemma3Model.get_image_features()`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L794).

#### `ImageTextToTextExportableModule`
A wrapper of `torch.nn.Module` for `image-text-to-text` task. Provides `export()` API that generates an `ExportedProgram`. It will be consumed by `xnnpack.py` recipe to generate ExecuTorch program.

### Custom Implementations
These are mostly copied from `optimum-executorch`. We put them here so that they can be reused by `integrations.py` and `xnnpack.py` recipe.

- **Custom KV Cache**: Optimized key-value cache implementations for ExecuTorch
- **Custom SDPA**: Scaled Dot-Product Attention optimizations
- **XNNPACK Integration**: Lower to XNNPACK backend for optimized inference on CPU

### Utilities

- Configuration saving and constant method generation
- Model metadata extraction
- Export helper functions

## Usage

```python
from transformers import PretrainedConfig
from executorch.extension.llm.optimum.image_text_to_text import load_image_text_to_text_model
from executorch.extension.llm.optimum.xnnpack import export_to_executorch_with_xnnpack
from executorch.extension.llm.optimum.modeling import ExecuTorchModelForImageTextToTextCausalLM

model_id = "google/gemma-3-4b-it"

module = load_image_text_to_text_model(
    model_id,
    use_custom_sdpa=True,
    use_custom_kv_cache=True,
    qlinear=True,
    qembedding=True,
)
model = export_to_executorch_with_xnnpack(module)
et_model = ExecuTorchModelForImageTextToTextCausalLM(model, PretrainedConfig.from_pretrained(model_id))
```

## Testing

Run tests with:
```bash
python -m pytest extension/llm/optimum/test/
```
