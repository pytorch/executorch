# Qwen3-VL — Vision-Language Model for ExecuTorch

ExecuTorch export and runtime for [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
a 2.2B parameter vision-language model. Export produces a single `.pte` with three methods:
`vision_encoder`, `text_decoder`, and `token_embedding`.

## Prerequisites

- ExecuTorch installed from source (with `EXECUTORCH_BUILD_KERNELS_QUANTIZED=ON`)
- [optimum-executorch](https://github.com/huggingface/optimum-executorch) (`pip install optimum-executorch`)
- [transformers](https://github.com/huggingface/transformers) (`pip install transformers`)

## Quick Start

### 1. Export (XNNPACK)

Uses `optimum-executorch` to export the model directly from HuggingFace:

```bash
optimum-cli export executorch \
  --model "Qwen/Qwen3-VL-2B-Instruct" \
  --task "multimodal-text-to-text" \
  --recipe "xnnpack" \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear "8da4w" \
  --qlinear_group_size 32 \
  --qlinear_encoder "8da4w,8da8w" \
  --qlinear_encoder_group_size 32 \
  --qembedding "8w" \
  --qembedding_encoder "8w" \
  --dtype "float32" \
  --output_dir="qwen3/Qwen3-VL-2B-Instruct-xnnpack"
```

### 2. Run

The runtime script uses PyTorch eager for the vision encoder (Conv3d is not yet
supported in the ExecuTorch portable runtime) and the exported `.pte` for text
decoding:

```bash
python examples/models/qwen3_vl/run_qwen3_vl.py \
  --model_path qwen3/Qwen3-VL-2B-Instruct-xnnpack/model.pte \
  --image_path /path/to/image.jpg \
  --prompt "What is in this image?" \
  --max_new_tokens 200
```

## Exported Methods

The vision encoder input shape depends on the image used during export
(positions are pre-computed for a specific patch grid). The shapes below
are for the default sample image (1000×667):

| Method | Input | Output |
|--------|-------|--------|
| `vision_encoder` | pixel_values `(2604, 1536)` | image_embeds `(651, 2048)` |
| `text_decoder` | embeds `(1, seq, 2048)` + cache_position `(seq,)` | logits `(1, seq, 151936)` |
| `token_embedding` | token_ids `(1, seq)` | embeds `(1, seq, 2048)` |

## Quantization

| Component | Config | Why |
|-----------|--------|-----|
| LLM decoder | `8da4w` (int8 dynamic act + int4 weight, group_size=32) | Best speed/quality tradeoff |
| Vision encoder | `8da4w,8da8w` (mixed 4w/8w linears) + `8w` embeddings | Preserves visual quality |
| Embedding | `8w` (int8 weight-only) | Large vocab (151K tokens) |

Quantized model size: ~1.4 GB (down from ~4.4 GB bf16).

## Architecture

```
pixel_values (N_patches, 1536)          [1536 = 3×2×16×16 flattened 3D patch]
  → PatchEmbed (Conv3d)
  → 32× ViT Blocks (1280-dim, 16 heads, M-RoPE)
  → Merger (4:1 spatial merge, 1280 → 2048)
  → image_embeds (N_merged, 2048)

Text tokens → token_embedding → (1, seq, 2048)

[image_embeds ∥ text_embeds] → interleave by token position
  → 28× Qwen3 Decoder Layers (2048-dim, 16 heads, GQA 8 KV, QK-norm)
  → logits (1, seq, 151936)
```

## Export Details

The `optimum-executorch` export handles three Qwen3-VL-specific concerns:

- **M-RoPE vision positions**: The vision encoder computes positions via
  data-dependent ops (`torch.linspace`, `repeat_interleave`) that `torch.export`
  cannot trace. These are pre-computed eagerly and stored as constants in the
  exported graph.

- **M-RoPE text decoder hook**: During text decoder export, `position_ids` are
  injected via a forward pre-hook to avoid the `get_rope_index` code path that
  requires `input_ids` (not available when exporting with `inputs_embeds`).

- **Conv3d in vision encoder**: The 3D patch embedding Conv3d is exported into
  the `.pte` but the ExecuTorch portable `aten::convolution.out` kernel does not
  yet support 5D inputs. The runtime script works around this by running the
  vision encoder through the HF model in PyTorch eager mode.
