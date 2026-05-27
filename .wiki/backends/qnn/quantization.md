---
title: QNN Quantization Guide
category: QUANTIZATION
backends: [QNN]
socs: [SM8450, SM8550, SM8650, SM8750, SA8255, SA8295, SA8797, SXR2230P]
last_validated: 2026-04-05
source_issues: [1182, 5929, 6846, 9127, 10226, 12747, 13092, 14032, 14402, 15410, 15954, 15998, 16013, 16427, 16488, 16615, 16690, 17296, 18280, 18571]
---

# QNN Quantization Guide

## Quantization Schemes

QNN ExecuTorch uses **weight-activation quantization** (not weight-only). All inference is done in the integer domain. [Source: #14032, #16013]

| Scheme | Description | Best For | Min Arch |
|--------|-------------|----------|----------|
| `use_8a8w` | 8-bit activation, 8-bit weight | Vision models, non-LLMs | V68 |
| `use_16a8w` | 16-bit activation, 8-bit weight | LLMs on V68/V69, accuracy-critical | V68 |
| `use_16a4w` | 16-bit activation, 4-bit weight (per-channel) | LLMs (good accuracy/speed tradeoff) | V68 |
| `use_16a4w_block` | 16-bit activation, 4-bit weight (block/LPBQ) | LLMs on V73+ (best perf) | **V73** |
| `use_16a16w` | 16-bit activation, 16-bit weight | High accuracy requirements | **V73** (for layer_norm, matmul) |

[Source: #15410, #16427, #12747]

## Recommended Recipes by Model Family

### LLMs (Llama, Qwen, SmolLM)

**On V73+ (SM8550+)**: Use `16a4w_block` with group_size=32 — this is the default and best-performing option.
```python
ptq = QuantDtype.use_16a4w_block
group_size = 32
```

**On V68/V69 (SA8295, SM8450, SXR2230P)**: Use `16a8w` per-channel with `annotate_kv_8bit`:
```python
ptq = QuantDtype.use_16a8w
group_size = None
custom_quant_annotations = [annotate_kv_8bit]
```
[Source: #15410, #15954, #16690]

**Why not 8a8w for LLMs?** 8-bit activation is insufficient for LLM activations which are very sparse. 16-bit activation is strongly recommended. [Source: #15954, #16013]

**Memory-bound vs compute-bound**: At 0.6B scale, models are memory-bound so `16a4w` is optimal. For very small models (SmolLM2 135M), `16a8w` may be faster due to lower dequantization overhead. [Source: #16013]

### Vision Models (DeepLab, Inception, ViT, YOLO)

Use `8a8w` with per-channel weight quantization for convolutions:
```python
quantizer = make_quantizer(quant_dtype=QuantDtype.use_8a8w)
```
[Source: #1182, #12134]

### Audio Models (wav2letter)

`8a8w` is the starting point. Some ops may not fully delegate. [Source: #7634]

## Per-Layer Quantization (Mixed Precision)

Use `add_regex` and `add_node_target` to apply different quantization per layer:

```python
recipe = (
    QuantRecipe(QuantDtype.use_16a4w_block, False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32)})
    # Keep conv2d per-channel (critical for accuracy)
    .add_node_target(
        {torch.ops.aten.conv2d.default},
        QuantDtype.use_16a8w, False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_CHANNEL,
    )
    # Protect sensitive layers
    .add_regex(
        {r"layers\..*\.feed_forward\.w2_conv"},
        QuantDtype.use_16a8w, False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_CHANNEL,
    )
)
```

**conv2d MUST use per-channel quantization** — per-tensor causes significant accuracy loss because weights have large variance across channels. [Source: #15954]

## Critical Quantization Rules

### conv2d requires per-channel
Conv2d weights typically have large variance across channels. Per-channel quantization is critical for maintaining accuracy. [Source: #15954]

### Sensitive layers need higher precision
For LLMs, `down_proj` and `lm_head` layers are most sensitive. Use `16a8w` per-channel for these even when rest of model uses `16a4w`. [Source: #14985, #17948]

```python
.add_regex(
    {r"output\.conv"},
    QuantDtype.use_16a8w, False,
    act_observer=MinMaxObserver,
    granularity=QuantGranularity.PER_CHANNEL,
)
```

### tanh requires special encodings in 16-bit
The `tanh` op requires fixed-point encodings in 16-bit quantization. A custom annotator is needed. [Source: #12747]

### Calibration affects accuracy at different seq_len
When changing `max_seq_len`, the calibration range changes. A model calibrated at `max_seq_len=1024` may produce wrong results at `max_seq_len=512`. Use `--tasks wikitext --limit 1` for stable calibration. [Source: #16615]

## Common Quantization Errors

### InsertIOQDQ pass failure
```
Exception: An error occurred when running the 'InsertIOQDQ' pass
after the following passes: ['FoldQDQ', 'InsertRequantize']
```
**Cause**: Op validation failure for the target SoC. Typically `layer_norm` or `matmul` not supported at the requested precision on V68. [Source: #16427]

**Fix**: Use custom annotations to downgrade unsupported ops:
```python
quantizer.add_custom_quant_annotations((annotate_kv_8bit,))
```

### KeyError in quantizer
```
KeyError: 'aten.native_layer_norm.default'
```
**Cause**: Missing annotation for the op in QnnQuantizer. [Source: #1182]

**Fix**: Check if the op needs custom annotation or if the quantizer version supports it.

### Op validation failed 3110
```
[ERROR] [Qnn ExecuTorch]: QnnBackend_validateOpConfig failed 3110
[ERROR] [Qnn ExecuTorch]: Failed to validate op X with error 0xc26
```
**Cause**: The op's quantization configuration is incompatible with the target arch. [Source: #12747, #17296]

**Fix**: Check the QNN Op Def Supplement (search for the op name in the Qualcomm QNN SDK documentation) for supported quantization configurations per op and HTP architecture.

### Segfault during 8a8w export
Using `8a8w` for LLMs can cause segfaults during compilation. This is a known issue being investigated. [Source: #16013]

## Calibration Best Practices

1. **Use task-based calibration** for LLMs: `--tasks wikitext --limit 1` provides diverse calibration data [Source: #16615]
2. **Include special tokens** in calibration data for instruct models:
   ```
   --calibration_data "<|start_header_id|>system<|end_header_id|>..."
   ```
   [Source: #5929]
3. **Calibration length matters**: Calibrating at `max_seq_len=512` vs `1024` produces different quantization ranges. Match calibration to deployment settings. [Source: #16615]

## SpinQuant Support

SpinQuant (rotation-based quantization) is supported for LLMs and can improve accuracy:
- Enable via model config: `r1 = True` (R1 rotation)
- Can be combined with SeqMSE for further optimization [Source: #15954, #9127]

## Verifying Quantization

Save the quantized model as `.pt2` to inspect with Netron or Model Explorer:
```python
captured_model = torch.export.export(model, inputs, strict=False)
torch.export.save(captured_model, "my_model.pt2")
```
This helps identify missing QDQ patterns and dtype mismatches. [Source: #12747]

## Skipping Quantization for Specific Nodes

Use `skip_node_op_set` to keep certain ops in FP16:
```python
from executorch.backends.qualcomm.utils.utils import skip_annotation
skip_annotation(quantizer, node_name_list)
```
Not recommended for HTP performance — fixed-point is generally faster than FP16 on HTP. [Source: #14032]

## LPBQ (Low Precision Block Quantization) Details

LPBQ is QNN's block-wise 4-bit quantization scheme (`use_16a4w_block`). Key details:
- Requires V73+ (SM8550 or newer)
- Uses block_size typically `(1, 32)` — each block of 32 weights shares a scale factor
- The `group_size` parameter in model config maps to block_size
- Provides best latency/accuracy tradeoff for LLMs on V73+ devices
[Source: #16488, #15410]

## Fake Quantized Model Accuracy Check

After `convert_pt2e`, the resulting fake-quantized model should produce similar outputs to the original float model. If the fake-quantized model already produces bad results, the issue is in quantization (calibration, precision choice), not in the QNN backend compilation. [Source: #13092]

```python
# Verify fake quant model before compiling to QNN:
quantized_model = convert_pt2e(prepared_model)
fake_quant_output = quantized_model(*sample_inputs)
float_output = original_model(*sample_inputs)
# Compare outputs — if divergent, fix quantization first
```

## CPU Utilization Differences Between Models

On the same device, smaller models (Qwen3-0.6B) may show higher CPU utilization than larger models (Qwen3-1.7B) because the smaller model spends proportionally more time in CPU-side token processing relative to HTP inference. This is expected behavior. [Source: #15998]

## See Also

- [SoC Compatibility Matrix](soc-compatibility.md) — V68/V69 quantization constraints, arch-specific limitations
- [QNN Debugging Guide](debugging.md) — Profiling quantized models, error diagnosis
- [QNN Known Issues](known-issues.md) — Gibberish output, compilation failures
- [General Quantization Recipes](../../quantization/recipes.md) — Cross-backend quantization guidance
- [Quantization Debugging](../../quantization/debugging.md) — Accuracy debugging after quantization
