# quant/

Quantization framework: **recipe → quantize → pack**.

## Files

| File | Concern | Depends on |
|---|---|---|
| `recipe.py` | **Policy** — what to quantize, what precision, which layers | nothing |
| `quantize.py` | **Computation** — produces torchao subclass tensors | recipe, torchao |
| `pack.py` | **Packing dispatch** — `pack_model` (bulk) and `pack_one` (streaming) | — |
| `pack_cuda.py` | **CUDA packing** — converts Int4Tensor to tinygemm format | pack |
| `gguf.py` | **GGUF import** — unpacks Q4_K/Q6_K blocks to torchao subclasses | torchao |

## Data flow

```
QuantRecipe → quantize_model() → state_dict{Int4Tensor, IntxUnpackedToInt8Tensor, Tensor} → safetensors → state_dict → pack_model() → runtime model
```

Quantized weights are stored as torchao tensor subclasses:
- **Int4Tensor** — 4-bit weights (nibble-packed qdata + transposed scale/zero_point)
- **IntxUnpackedToInt8Tensor** — 8-bit weights (int8 qdata + scale + zero_point)

These are the canonical interchange formats from torchao. Everything left
of `save()` is backend-agnostic. Everything right is backend-specific.

## Adding a new backend

Write a `pack_<backend>.py` with per-module packers and a default registry:

```python
def pack_linear_for_metal(module, weights): ...
DEFAULT_METAL_PACKERS = {nn.Linear: pack_linear_for_metal}
```

Call `pack_model(model, state_dict, packers=DEFAULT_METAL_PACKERS)`.
No changes to recipe or quantize.

## On-disk format

Uses torchao's safetensors integration (`torchao.prototype.safetensors`).
Each tensor subclass is decomposed into its inner tensors
(e.g., `layer._weight_qdata`, `layer._weight_scale`) plus JSON metadata
recording the subclass type and attributes. Plain tensors are stored as-is.
The format is compatible with torchao's `save_pretrained` / `load_pretrained`.

## TODO

- `pack_metal.py` — Metal backend packer.
- `pack_mlx.py` — MLX backend packer.
- `gguf.py` — extend with Q5_K, Q8_0 GGUF quant types.
- Upstream `Int4TilePackedTo4dTensor.from_int4_tensor()` to torchao
  to replace the manual conversion in `pack_int4_for_cuda`.
