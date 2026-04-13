# Quantization

The MLX backend supports weight-only quantization via [TorchAO](https://github.com/pytorch/ao) for reducing model size and improving inference performance, particularly for LLMs on Apple Silicon. Quantization is applied to the eager model in-place **before** `torch.export()`.

## `quantize_`

The MLX backend uses TorchAO's [`quantize_`](https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html) API under the hood. You can call it directly for full control over quantization configs and granularity. The key TorchAO configs are:

- [`IntxWeightOnlyConfig`](https://docs.pytorch.org/ao/main/generated/torchao.quantization.IntxWeightOnlyConfig.html) — for INT2/INT4/INT8 weight-only quantization with per-group granularity (group sizes 32, 64, 128)
- [`ExportableNVFP4Config`](https://github.com/pytorch/executorch/blob/main/extension/llm/export/nvfp4.py) — for NVFP4 weight-only quantization

```python
import torch
from torchao.quantization.quant_api import quantize_, IntxWeightOnlyConfig
from torchao.quantization.granularity import PerGroup

# INT4 weight-only quantization for linear layers (group_size=32)
quantize_(
    model,
    IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
    filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear),
)

# INT8 weight-only quantization for embedding layers (group_size=128)
quantize_(
    model,
    IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerGroup(128)),
    filter_fn=lambda m, fqn: isinstance(m, torch.nn.Embedding),
)
```

## `quantize_model_`

For convenience, ExecuTorch provides `quantize_model_` which wraps `quantize_` with sensible defaults for common LLM quantization configurations:

```python
from executorch.extension.llm.export.quantize import quantize_model_

# Quantize linear layers with INT4, embedding layers with INT8
# Note: 8w defaults to per-axis grouping, which MLX does not support.
# Always pass an explicit group size when using 8w with MLX.
quantize_model_(model, qlinear_config="4w", qembedding_config="8w", qembedding_group_size=128)
```

### Supported configs

| Config | Description |
|--------|-------------|
| `4w` | INT4 weight-only quantization (per-group) |
| `8w` | INT8 weight-only quantization (per-group) |
| `nvfp4` | NVIDIA FP4 weight-only quantization |

These can be applied independently to linear layers and embedding layers.

### Using the LLM Export Script

The simplest way to export a quantized model is via the `export_llm_hf` script, which calls `quantize_model_` internally:

```bash
# INT4 quantization for both linear and embedding layers
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_int4.pte \
    --use-custom-sdpa \
    --use-custom-kv-cache \
    --qlinear 4w \
    --qembedding 4w

# INT8 quantization for linear layers only
# Note: --qlinear-group-size is required for 8w (default is per-axis, which MLX does not support)
python -m executorch.backends.mlx.examples.llm.export_llm_hf \
    --model-id "unsloth/Llama-3.2-1B-Instruct" \
    --output llama_int8.pte \
    --use-custom-sdpa \
    --use-custom-kv-cache \
    --qlinear 8w \
    --qlinear-group-size 128
```

### CLI Quantization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--qlinear` | None | Quantization for linear layers (`4w`, `8w`, `nvfp4`) |
| `--qembedding` | None | Quantization for embedding layers (`4w`, `8w`, `nvfp4`) |
| `--qlinear-group-size` | Depends on config | Group size for linear layer quantization (32, 64, or 128). Defaults to 32 for `4w`, 16 for `nvfp4`. **Required for `8w`** (default is per-axis, which MLX does not support). |
| `--qembedding-group-size` | Depends on config | Group size for embedding layer quantization (32, 64, or 128). Defaults to 32 for `4w`, 16 for `nvfp4`. **Required for `8w`** (default is per-axis, which MLX does not support). |
| `--no-tie-word-embeddings` | False | Disable re-tying lm_head to embedding after quantization |
