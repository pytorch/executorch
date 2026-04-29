# quant/

Packing-agnostic quantization framework: **recipe → quantize → serialize → pack**.

## Files

| File | Concern | Depends on |
|---|---|---|
| `recipe.py` | **Policy** — what to quantize, what precision, which layers | nothing |
| `quantize.py` | **Computation** — produces canonical weights from fp weights | recipe, torchao |
| `serialize.py` | **Data format** — saves/loads canonical weights to safetensors | recipe |
| `pack.py` | **Packing dispatch** — walks model, dispatches to per-module packers | serialize |
| `pack_cuda.py` | **CUDA packing** — converts canonical to tinygemm/intx runtime format | pack, serialize |

## Data flow

```
QuantRecipe → quantize_model() → CanonicalQuantizedWeight → save() → file → load() → CanonicalQuantizedWeight → pack_model() → runtime model
```

`CanonicalQuantizedWeight` is the interchange point — int8 qdata + bf16
scale + optional zero + config. Everything left of it is backend-agnostic.
Everything right is backend-specific.

## Adding a new backend

Write a `pack_<backend>.py` with per-module packers and a default registry:

```python
def pack_linear_for_metal(module, weights): ...
DEFAULT_METAL_PACKERS = {nn.Linear: pack_linear_for_metal}
```

Call `pack_model(model, quantized, unquantized, packers=DEFAULT_METAL_PACKERS)`.
No changes to recipe, quantize, or serialize.

Things to consider:

- **Recipes may need to be backend-aware.** Each backend's kernels have
  different constraints (e.g., Metal's `fpa4w` is INT4-only — no INT8 linear
  kernel, so the sensitive recipe's 8-bit edge layers would need to be INT4
  or dequantized to bf16). Define per-backend recipes or validate recipe
  compatibility at pack time.
- **Source transforms before packing.** Some backends replace model modules
  (e.g., MLX swaps `FusedMoEExperts` → `SwitchMLP`, Metal swaps to
  `MetalMoEExperts`). These transforms change the module types that
  packers dispatch on, so they must run before `pack_model()`. For dense
  models (no MoE) this is not needed.
- **Embedding quantization.** Not all backends have a quantized embedding
  gather kernel. The packer can dequantize to bf16 at load time — the
  disk savings from the canonical format still apply.

## Adding a new model

1. Define a `QuantRecipe` with rules for the model's FQN patterns.
2. If the model has custom module types (e.g., `FusedMoEExperts`), write a
   per-module packer and extend the packers dict:
   ```python
   packers = {**DEFAULT_CUDA_PACKERS, FusedMoEExperts: pack_moe_experts}
   ```
3. No changes to the quant package itself.

## On-disk format

Safetensors with a `format_version` in the header. Per quantized weight:
`{fqn}.qdata` (int8, nibble-packed for 4-bit), `{fqn}.scale` (bf16),
optionally `{fqn}.zero` (bf16). Header JSON records bits, group_size,
symmetric, and method per weight. Unquantized weights stored as-is.

## TODO

- `pack_metal.py` — Metal backend packer. Convert canonical INT4 to
  `UIntxWeightOnlyConfig` subclass (torchao experimental) for the
  `torchao::_linear_fp_act_4bit_weight` kernel. For MoE models, pack
  expert weights into Metal's `gather_qmv` format (asymmetric, unsigned
  INT4 with scale + bias buffers).

- `pack_mlx.py` — MLX backend packer. Convert canonical INT4 to
  `IntxWeightOnlyConfig` subclass for the `mlx::gather_qmm` kernel.
  For MoE models, stack per-expert weights into `SwitchLinear` format.

- `gguf.py` — read a GGUF file and convert to `CanonicalQuantizedWeight`
  dicts, enabling `load() → pack_model()` from community-quantized GGUF
  checkpoints without re-quantizing from bf16. Maps GGUF quant types
  (Q4_K, Q6_K, Q8_0, etc.) to `QuantConfig` and unpacks super-blocks
  into the canonical qdata + scale + zero layout. For CUDA packing,
  Q6_K would be widened to 8-bit (`pack_int8_for_cuda`) since there is
  no 6-bit CUDA kernel — lossless, ~33% more memory than true 6-bit.
