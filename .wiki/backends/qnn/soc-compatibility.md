---
title: QNN SoC Compatibility Matrix
category: BACKEND_CONSTRAINT
backends: [QNN]
socs: [SM8350, SM8450, SM8475, SM8550, SM8650, SM8750, SM8845, SM8850, SA8255, SA8295, SA8797, SSG2115P, SSG2125P, SXR1230P, SXR2230P, SXR2330P, QCM6490, QCS9100, SAR2230P, SW6100]
last_validated: 2026-04-05
source_issues: [1176, 4973, 8454, 13216, 14032, 15410, 15954, 16427, 16465, 16535, 16690, 17296, 18280]
---

# QNN SoC Compatibility Matrix

## SoC to HTP Architecture Mapping

| SoC Model | HTP Arch | Device Examples | Min QNN SDK |
|-----------|----------|-----------------|-------------|
| SM8350 | V68 | 888 | 2.14+ |
| SA8295 | V68 | Qualcomm automotive platforms | 2.14+ |
| QCM6490 | V68 | IoT platforms | 2.37+ |
| SM8450 | V69 | Galaxy S22, 8 Gen 1 | 2.14+ |
| SM8475 | V69 | 8+ Gen 1 | 2.14+ |
| SXR2230P | V69 | Meta Quest 3 | 2.37+ |
| SM8550 | V73 | Galaxy S23, 8 Gen 2 | 2.23+ |
| SA8255 | V73 | Qualcomm automotive | 2.37+ |
| SSG2115P / SSG2125P | V73 | XR / smart-glasses platforms | 2.37+ |
| SXR1230P | V73 | XR platform | 2.37+ |
| QCS9100 | V73 | Automotive/industrial | 2.37+ |
| SM8650 | V75 | Galaxy S24, 8 Gen 3 | 2.23+ |
| SM8750 | V79 | Galaxy S25, 8 Elite | 2.37+ |
| SXR2330P | V79 | XR platform | 2.37+ |
| SA8797 | V81 | Automotive (16 MB VTCM) | 2.42+ |
| SM8845 | V81 | Mobile | 2.42+ |
| SM8850 | V81 | Mobile (8 MB VTCM) | 2.42+ |
| SAR2230P | V81 | XR platform | 2.42+ |
| SW6100 | V81 | Platform variant | 2.42+ |

HTP arch / VTCM size are defined in `backends/qualcomm/serialization/qc_schema.py` (`_soc_info_table`). CQ8750S (soc_id=705) is **not** in the mainline `QcomChipset` enum as of this writing — treat it as requiring an upstream patch until it is merged. [Source: #1176, #4973, #16535, #16465]

## Feature Support by Architecture

| Feature | V68 | V69 | V73 | V75 | V79 | V81 |
|---------|-----|-----|-----|-----|-----|-----|
| 8a8w quantization | Yes | Yes | Yes | Yes | Yes | Yes |
| 16a8w quantization | Yes | Yes | Yes | Yes | Yes | Yes |
| 16a4w per-channel | Yes | Yes | Yes | Yes | Yes | Yes |
| 16a4w block (LPBQ) | **No** | **No** | Yes | Yes | Yes | Yes |
| 16a16w layer_norm | **No** | **No** | Yes | Yes | Yes | Yes |
| 16-bit matmul (2nd input) | **No** | **No** | Yes | Yes | Yes | Yes |
| Weight sharing | **No** | **No** | Yes | Yes | Yes | Yes |
| FP16 graph | **No** | Partial | Yes | Yes | Yes | Yes |
| Shared buffer | Yes | Yes | Yes | Yes | Yes | Yes |
| Multi-core NPU | N/A | N/A | N/A | N/A | N/A | Yes |

[Source: #15410, #16427, #17296, #18280, #14032]

## V68 (SA8295) Limitations — Critical

V68 is the most constrained architecture. Many default quantization recipes fail on V68:

### No LPBQ (Low Precision Block Quantization)
Block-wise 4-bit quantization (`use_16a4w_block`) requires V73+. On V68, use per-channel `use_16a8w` instead. [Source: #15410]

```python
# V68-compatible config for Qwen3-0.6B
class Qwen3_0_6B_V68(LLMModelConfig):
    ptq = QuantDtype.use_16a8w  # NOT use_16a4w_block
    group_size = None            # No block quantization
    custom_quant_annotations = [annotate_kv_8bit]  # 8-bit KV cache
```

### No 16-bit matmul second input
Matmul ops with 16-bit second input (including KV cache) require V73+. Use `annotate_kv_8bit` to force 8-bit KV cache. [Source: #15410, #17296]

```python
# Add to custom_quant_annotations for V68
from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_kv_8bit
quantizer.add_custom_quant_annotations((annotate_kv_8bit,))
```

### No 16a16w layer_norm
Layer norm with 16-bit weights and activations requires V73+. On V68, annotate layer_norm with 8a8w. [Source: #17296, #18280]

```python
# Patch for V68: annotate layer_norm as 8a16w
def annotate_for_v68(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.native_layer_norm.default:
            # Use 8-bit weights for layer_norm on V68
            ...
```

See the full patch in [#18280] for complete V68 annotation.

### Error signature for V68 arch violations
```
[ERROR] [Qnn ExecuTorch]: [4294967295] has incorrect Value 68, expected >= 73.
[ERROR] [Qnn ExecuTorch]: QnnBackend_validateOpConfig failed 3110
```
This means the op requires a higher HTP arch than V68. [Source: #15954, #17296, #18280]

## V69 (SM8450, SXR2230P) Limitations

V69 shares most V68 limitations:
- No LPBQ support
- No 16-bit matmul 2nd input (same constraint as V68; use `annotate_kv_8bit` for LLM KV cache) [Source: #15410, #16690, #17296]
- Weight sharing not supported [Source: #15387]
- For LLMs, use `annotate_kv_8bit` in quantization recipe [Source: #16690]

## V73+ (SM8550, SA8255) Capabilities

V73 is the minimum architecture where default LLM quantization recipes work without manual overrides. V68/V69 can run LLMs but require custom recipe configuration. [Synthesis — derived from #15410, #16690, #14032]
- Full 16-bit matmul support [Source: #15410]
- LPBQ (block-wise quantization) support [Source: #15410]
- 16a16w layer_norm support [Source: #17296, #18280]
- Weight sharing support (reduces `.pte` file size) [Source: #14032]

## Identifying Your SoC

### From device
```bash
adb shell getprop ro.soc.model    # e.g., SM8650
adb shell cat /sys/devices/soc0/soc_id  # numeric ID
```

### From error logs
The QNN runtime logs the detected SoC:
```
[INFO] [Qnn ExecuTorch]: Get soc info for soc model 57.  # SM8650
[INFO] [Qnn ExecuTorch]: Get soc info for soc htp arch 75.
```
SoC model IDs are defined in `backends/qualcomm/serialization/qc_schema.py`. [Source: #1176]

## Adding New SoC Support

To add a new SoC, modify these files (see PR #16694 for SA8797 as example):

1. `backends/qualcomm/serialization/qc_schema.py` — Add to `QcomChipset` enum (SoC table is entirely in this Python file)
2. `backends/qualcomm/serialization/qc_compiler_spec.fbs` — Add to flatbuffer schema
3. Push the correct `libQnnHtpV{XX}Stub.so` to device

[Source: #1176, #16535]

## SoC-Specific Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `has incorrect Value 68, expected >= 73` | Op requires V73+ arch | Use V68-compatible quantization [Source: #17296] |
| `Request feature arch with value 75 unsupported` | `.pte` compiled for wrong SoC | Recompile with correct `-m` flag [Source: #11100] |
| `No Snapdragon SOC detected` | SoC ID not in ExecuTorch's table | Add SoC ID mapping or upgrade QNN SDK [Source: #16465] |
| `graph requires estimated allocation of X KB, limit is Y KB` | Model too large for HTP PD memory | Increase `num_sharding` or reduce model/seq_len [Source: #15954, #17782] |
| `Failed to find available PD` | All HTP PDs exhausted | Reduce number of context binaries or shard count [Source: #18410, #14985] |

## Unsupported SoCs

| SoC | HTP Arch | Why Unsupported |
|-----|----------|-----------------|
| SA8155 | V66 | QNN-HTP does not support V66. Would need QNN-DSP backend (not available in ExecuTorch). [Source: #1176] |

## Screen On/Off Performance Difference

On SM8650 (and potentially other SoCs), QNN inference performance differs significantly between screen-on and screen-off states due to thermal throttling and clock frequency changes. This is a device-level behavior, not an ExecuTorch issue. [Source: #13216]

## QNN SDK Version Compatibility

- Context binaries are **not forward compatible** — SDK version on host must match or be compatible with device [Source: #4155]
- Always set `QNN_SDK_ROOT` and `LD_LIBRARY_PATH` consistently when switching SDK versions [Source: #1430]
- Some SoCs require minimum SDK versions (e.g., SA8797 needs v2.42+, CQ8750S needs v2.43+) [Source: #16535, #16465]

## SM8750 Android APK Setup

When building Android apps targeting SM8750, ensure the V79 stub/skel libraries are included. Early versions of the APK build scripts didn't include V79 libraries. Use `backends/qualcomm/scripts/build.sh` which handles this automatically. [Source: #8454]

## See Also

- [QNN Quantization Guide](quantization.md) — Per-SoC quantization recipes, mixed precision
- [QNN Debugging Guide](debugging.md) — SoC detection in logs, arch mismatch errors
- [QNN Known Issues](known-issues.md) — Active issues per SoC
