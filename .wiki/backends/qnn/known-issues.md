---
title: QNN Known Issues
category: DEBUGGING
backends: [QNN]
socs: [SM8450, SM8550, SM8650, SM8750, SA8255, SA8295, SXR2230P]
last_validated: 2026-04-15
source_issues: [4075, 5929, 7550, 7634, 8139, 9084, 10226, 10580, 10895, 11034, 11307, 12161, 13608, 13611, 13612, 13629, 14032, 14048, 14049, 14050, 14052, 14402, 14652, 14985, 15410, 15732, 15734, 16013, 16123, 16310, 16413, 16465, 16557, 16616, 16999, 17136, 17732, 18571, 18795, 18806, 18812, 18862]
---

# QNN Known Issues

## Active Issues with Workarounds

### Gibberish/Repetitive Output from LLMs

**Symptoms**: Model outputs random text, repeated characters like ")" or "sp", or multilingual gibberish. [Source: #5929, #11034, #14402, #15410, #18571]

**Root Causes & Fixes**:

1. **Wrong quantization for SoC**: V68 devices fail with default `16a4w_block` recipes. Use `16a8w` per-channel with `annotate_kv_8bit`. [Source: #15410]

2. **Missing chat template**: Instruct models (Qwen3, Llama3 Instruct) require chat template. Recent mainline auto-applies it. [Source: #14402]

3. **Wrong runner**: Using `llama_main` instead of `qnn_llama_runner` for QNN-exported models. [Source: #11100]

4. **KV cache bit mismatch in Android app**: The JNI layer may use different KV cache configuration. Fix: apply PR #15258. [Source: #18571]

5. **Insufficient calibration**: Use `--tasks wikitext --limit 1` for better calibration data. [Source: #5929]

### x86 Emulator Does Not Support Weight Sharing

Running QNN LLM models on x86 emulator fails if weight sharing is enabled (default for V73+). [Source: #14032]

**Workaround**: Disable weight sharing when targeting x86 emulator, or use `--enable_x86_64` flag which disables shared buffer.

### High Host Memory During Compilation

Compiling large models (Qwen3-1.7B+) can consume 100-117GB of host RAM. [Source: #14402, #17782]

**Workaround**: 
- Increase swap space
- Use smaller `max_seq_len` during compilation
- Memory optimization PR is in progress

### PTE File Size Larger Than Float Model (16a4w)

**Applies to: examples/models/llama/ path (deprecated)**

Using the old `examples/models/llama/export_llama` path produces oversized `.pte` files (e.g., 2.9GB for a 2.4GB float model). [Source: #10226]

**Fix**: Use `examples/qualcomm/oss_scripts/llama/llama.py` instead. Fixed by PR #12167.

### NumPy Version Incompatibility

NumPy >= 2.0 can cause `RuntimeError: Unable to cast Python instance of type <class 'numpy.ndarray'> to C++ type '?'` during QNN compilation. The root cause is the numpy 2.x C ABI break affecting pybind11 casts in `PyQnnManagerAdaptor`. [Source: #16557, #18795]

**Fix**: Use Python 3.12 with numpy < 2.0:
```bash
conda create -n executorch python=3.12
conda activate executorch
pip install numpy==1.26.4
# Rebuild executorch + QNN backend from source
```

**Note**: Python 3.13+ requires numpy >= 2.0, making it incompatible. Downgrading to numpy 2.2.6 may work in some configurations but is not a reliable fix. If you must use Python 3.13, rebuild PyQnnManagerAdaptor with `-DCMAKE_CXX_FLAGS="-DPYBIND11_DETAILED_ERROR_MESSAGES"` for better diagnostics.

### Float.NEGATIVE_INFINITY Not Supported in QNN Attention Masks

**Symptom**: Decode model produces gibberish/repetitive output (e.g., "otropicскоескоеское") while prefill model works correctly. Occurs when using custom KV-cache inference code (not `qnn_llama_runner`). [Source: #18812]

**Root Cause**: QNN HTP cannot represent `Float.NEGATIVE_INFINITY` in FP16 attention masks. The value is silently misrepresented, causing attention to attend to masked positions.

**Fix**: Use a large finite negative value instead of `Float.NEGATIVE_INFINITY`:
```kotlin
// WRONG — QNN cannot represent this
private const val CAUSAL_MASK_VALUE = Float.NEGATIVE_INFINITY

// CORRECT — use a large finite value
private const val CAUSAL_MASK_VALUE = -255.0f  // or -65535.0f
```

This applies to any custom inference code that constructs causal attention masks for QNN models. The `qnn_llama_runner` handles this internally. [Source: #18812]

### HTP Performance Mode Has No Effect on Decode Speed

**Symptom**: Setting `--htp_performance_mode` (burst=2, power_saver=4) changes HTP clock frequency and bandwidth but does not significantly change decode token rate for weight-memory-bound LLMs. [Source: #18806]

**Details**: Performance mode correctly affects power profile (verified via `coreVoltageCornerMin` in QNN verbose logs), but decode speed for LLMs is dominated by weight memory transfers, not compute. On SA8295 (V68) with Qwen3-0.6B 4-bit, burst vs power_saver shows ~1 tok/s difference (~37 vs ~36 tok/s).

**Configuration (AOT)**:
```python
htp_options.performance_mode = QnnExecuTorchHtpPerformanceMode.kHtpBalanced
# or kHtpPowerSaver, kHtpBurst, etc.
```

**Configuration (runtime, qnn_executor_runner only)**:
```bash
./qnn_executor_runner --model_path model.pte --htp_performance_mode 4
```

**Note**: `qnn_llama_runner` and `qnn_multimodal_runner` do not yet support the `--htp_performance_mode` runtime flag. Setting it at AOT via `htp_options.performance_mode` is the supported path. [Source: #18806]

### InsertIOQDQ Pass Failure with Certain Quantization Recipes

**Symptom**:
```
Exception: An error occurred when running the 'InsertIOQDQ' pass after the following passes: ['FoldQDQ', 'InsertRequantize']
```
Occurs during QNN export of certain models (e.g., Qwen2.5-0.5B on SA8295). [Source: #17732]

**Root Cause**: When a quantized node is directly consumed by the output node, the `InsertIOQDQ` pass attempts to insert a dequantize node based on `QCOM_ENCODING`, but not all encodings are covered in `q_dq_map`. The pass assumes the mapping always exists.

**Reported workaround (single source)**: guard the `InsertIOQDQ` pass so it skips dq insertion when the node's encoding has no mapping in its internal encoding→dq lookup. This is a local workaround, not an official fix. Track PR #18601 for the upstream resolution. [Source: #17732]

### SMMU FastRPC mmap Error on Large Weight Buffers (~467MB+)

**Symptom**:
```
[ERROR] SMMU fastrpc mmap error (err 1002)
```
Occurs during QNN context creation when the model's weight buffer exceeds ~467MB. Reported on SA8295 with InternVL-2B (24 shards). [Source: #18862]

**Status**: Open issue, no workaround confirmed. Likely a device-level SMMU mapping limit. Potential mitigations: increase `num_sharding`, use more aggressive weight quantization, or reduce model size.

### QNN Context Binary Limit (~50 Partitions)

Models with >50 context binaries can fail to load at runtime due to PD exhaustion. [Source: #14985]

**Workaround**: Reduce number of partitions by keeping more ops on the QNN side or reducing custom partitioning granularity.

### DMA-BUF Second Load Failure

Loading and unloading a QNN model, then loading it again in the same Android app session fails. [Source: #15732]

**Fix**: Update to mainline — fixed by PR #16000 which removed legacy preregistration code.

### Stable Diffusion via QAIHUB Flow is Broken

The QAIHUB-based Stable Diffusion flow (`examples/qualcomm/qaihub_scripts/stable_diffusion/`) produces noise images. The flow is being deprecated in favor of native QNN export. [Source: #14652, #16407]

**Workaround**: Use native `build_executorch_binary` to export SD components directly. Native SD support is planned.

### conv1d Performance Issue

Framework converts conv1d to unsqueeze + conv2d + squeeze. After layout transform, extra permute ops are inserted that can dominate execution time. [Source: #12537]

**Workaround**: Replace `nn.Conv1d` with `nn.Conv2d` in the model definition (unsqueezing weights manually).

### CQ8750S Device Not Recognized

CQ8750S (soc_id=705) is not in ExecuTorch's SoC table, causing "No Snapdragon SOC detected". Requires QNN SDK v2.43+. [Source: #16465]

**Workaround**: Wait for SDK v2.43 release or manually add the SoC ID mapping.

## Resolved Issues (Instructive)

### Conv1dToConv2d Pass IndexError

```
IndexError: tuple index out of range
```
Off-by-one in the argument count check inside the conv canonicalization pass. Fixed by PR #12297. [Source: #12161]

### linear Op With Dynamic Weight (split_with_sizes output)

Linear op fails when weight is not a static parameter but the output of another op (e.g., `split_with_sizes`). Fixed by using `get_tensor` instead of `get_parameter` in op_linear.py. [Source: #15734]

**Fix**: PR #16014.

### ViT Lowering Failure (Missing Layer Norm)

ViT requires custom quantizer annotation for `aten.scaled_dot_product_attention.default`. Enabled via PR #1442. [Source: #1182]

### Mutable Buffer / Weight-Only Quantization

Mutable buffers (e.g., `register_buffer` with in-place ops) cause linear op failure because buffer outputs don't have parameter values. QNN ExecuTorch does not support weight-only quantization. [Source: #4075, #14032]

### YOLOv9 Layout Transform Crash

`stack` op in `layout_transform.py` caused assertion failure for YOLO models. Fixed by removing stack from layout-sensitive op list. [Source: #16616]

### 16KB Page Size Alignment for Android

Android 15+ requires 16KB page alignment for shared libraries. Fixed in ExecuTorch 1.0+ releases. Build with `-Wl,-z,max-page-size=16384` if building from source. [Source: #11518]

### SM8650 Hardcoded in Partitioner

Early versions hardcoded SM8650 in the partitioner, causing failures on other SoCs. Fixed by making SoC configurable via `-m` flag (PR #5211). [Source: #4973]

## Version-Specific Notes

### ExecuTorch <= 0.4
- PTE size bug in `examples/models/llama/` path [Source: #10226]
- Limited SoC support

### ExecuTorch 1.0+
- 16KB alignment fix for Android
- Improved QNN quantizer with automatic op validation
- Better V68/V69 support with custom annotations

### QNN SDK Versions
- **v2.14**: Minimum supported, V68/V73/V75
- **v2.23+**: SM8550/SM8650 support
- **v2.37+**: SM8750, improved op coverage
- **v2.42+**: SA8797 (V81) support
- **v2.43+**: CQ8750S support
- **v2.44+**: Latest features

Context binaries are not forward-compatible across SDK versions. [Source: #1430, #4155]

### Transposed Conv2d with Dilation Incorrect

Transposed convolution with dilation > 1 may produce incorrect outputs on QNN backend. [Source: #13611]

**Workaround**: Avoid dilation > 1 in transposed conv2d, or validate outputs against CPU reference.

### Avg/Max Pool with ceil_mode=True Incorrect

Pooling ops with `ceil_mode=True` may produce incorrect outputs. [Source: #13612]

**Workaround**: Set `ceil_mode=False` and adjust padding manually if needed.

### All-Dim Reduction Ops IndexError

Reduction operators (sum, mean) applied across all dimensions fail with `IndexError: tuple index out of range`. [Source: #13608]

**Fix**: Fixed in mainline.

### Conformer/ConvNext/Swin/MaxViT Lowering Failures

Several vision transformer variants fail to lower on QNN:
- **Conformer**: Fails during lowering [Source: #14048]
- **ConvNext**: Fails during lowering [Source: #14049]
- **MaxViT**: Segfaults during lowering [Source: #14050]
- **Swin_v2**: Fails during lowering [Source: #14052]

These models may require custom annotations or op support additions.

### Missing `#include <unordered_map>` in rpc_mem.h

Building `qnn_llama_runner` fails with `no template named 'unordered_map'`. [Source: #11307]

**Fix**: Fixed in PR #11515 — update to latest ExecuTorch.

### KeyError: 'aten.alias_copy.default'

Using the old `examples/models/llama/export_llama` path may produce `KeyError: 'aten.alias_copy.default'` during partitioning. [Source: #10895]

**Fix**: Use the new flow at `examples/qualcomm/oss_scripts/llama/llama.py` instead.

### ModuleNotFoundError: 'executorch.backends.qualcomm.python'

```
ModuleNotFoundError: No module named 'executorch.backends.qualcomm.python'
```
**Cause**: PyQnn pybind libraries not copied to the correct location after build. [Source: #16310]

**Fix**: After building, copy the QNN pybind `.so` files from the build output into the `backends/qualcomm/python/` source directory so they are importable.

## Feature Requests Tracked

| Feature | Issue | Status |
|---------|-------|--------|
| Native Stable Diffusion support | #16407 | Planned |
| Multi-core NPU support | #16762 | PR #17090 in progress |
| QNN GPU backend | #5914 | Experimental (PR #12165) |
| Batch inference | #16413 | Not supported (batch=1 only) |
| Pre-trained MTP (Multi-Token Prediction) | #16413 | Not supported; lookahead decoding available |
| Public .pte repository | #11034 | In progress at HuggingFace |
| Heterogeneous QNN + XNNPACK | #13629 | Possible via partitioner — ops not delegated to QNN fall back to CPU/XNNPACK |
| Multi-LoRA support | #16999 | Not supported |
| Dynamic weight update / fine-tuning | #16123 | Not supported in QNN/XNNPACK delegates |
| Windows host QNN AOT | #17136 | Not supported (Linux/macOS only) |
| ETDump from qnn_llama_runner | #10580 | Feature request, not yet implemented |
