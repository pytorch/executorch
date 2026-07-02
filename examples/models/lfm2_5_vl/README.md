# LFM2.5-VL ExecuTorch Export

Export the LFM2.5-VL family to ExecuTorch as a single multi-method PTE compatible with the LLaVA C++ runner. Both checkpoints are supported and share the same export path:

- [LiquidAI/LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B) — text dim 2048
- [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) — text dim 1024

LFM2.5-VL is a **hybrid SSM+attention vision-language model** — 16 decoder layers alternating between short convolution blocks and full attention blocks, paired with a SigLIP ViT vision encoder.

## Architecture

Three named methods in one PTE (`D` = text hidden dim: 2048 for 1.6B, 1024 for 450M):

| Method | Input | Output |
|--------|-------|--------|
| `vision_encoder` | `[1, 3, 512, 512]` float32 NCHW pixels [0,255] | `[1, 256, D]` float32 |
| `token_embedding` | `[1, seq_len]` int64 token IDs | `[1, seq_len, D]` float32 |
| `text_decoder` | `([1, seq_len, D]` float32, `[seq_len]` int64) | `[1, 65536]` float32 |

## Export

```bash
# 1.6B (default)
python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \
    --model_dir LiquidAI/LFM2-VL-1.6B \
    --dtype fp32

# 450M — bundled config is auto-selected from --model_dir
python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \
    --model_dir LiquidAI/LFM2.5-VL-450M \
    --dtype fp32
```

With quantization (8da4w decoder + int8 embedding + float32 vision encoder):

```bash
python examples/models/lfm2_5_vl/export_lfm2_5_vl.py \
    --model_dir LiquidAI/LFM2.5-VL-450M \
    --quantize
```

The bundled architecture configs live in [config/](config/). Pass `--params /path/to/custom.json` to override.

### Required runner configuration

- Resize image to exactly 512×512
- Pass CHW float32 pixels in [0, 255] — normalization is baked into `vision_encoder`
- EOS token: `<|im_end|>` (ID 7), read from `get_eos_ids()` constant method
- Chat template: `<|startoftext|><|im_start|>user\n<image>{prompt}<|im_end|>\n<|im_start|>assistant\n`

## Key Implementation Notes

- `enable_dynamic_shape=False` — required to avoid `.item()` in RoPE's `get_freqs` during FakeTensor export
- `strict=False` — hybrid conv layers have buffer mutations not traceable in strict mode
- Text decoder exported before token embedding — `EmbeddingQuantHandler` mutates weights in-place
- Bilinear PE precompute with `antialias=True` — validated correct vs HF processor output
