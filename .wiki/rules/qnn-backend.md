---
title: QNN Backend Rules
category: BACKEND_CONSTRAINT
backends: [QNN]
last_validated: 2026-04-15
source_issues: [1176, 4973, 5929, 10226, 10895, 10966, 11100, 11807, 12161, 12537, 14032, 14402, 15410, 15954, 16013, 16415, 16427, 16465, 16535, 16557, 16615, 16690, 17296, 17755, 18280, 18410, 18812]
---

# QNN Backend — Critical Rules

## SoC Architecture Constraints

1. **V68 (SA8295) cannot use LPBQ/block quantization** (`use_16a4w_block`). Use `use_16a8w` per-channel instead. [#15410]

2. **V68/V69 cannot use 16-bit matmul 2nd input** (as of QNN SDK 2.x). This is enforced by the SDK's `QnnBackend_validateOpConfig` op-config check, not a silicon-level lockout — future SDKs may relax it. Add `annotate_kv_8bit` to custom annotations for all LLM recipes on V68/V69/SXR2230P. [#15410, #16690, #17296]

3. **V68/V69 cannot use 16a16w for layer_norm or matmul**. Annotate layer_norm as 8a16w on these archs. [#17296, #18280]

4. **Weight sharing requires V73+**. Disabled automatically for x86 emulator. [#14032]

5. **New SoC not recognized → "No Snapdragon SOC detected"**. Check `qc_schema.py` for SoC ID, may need newer QNN SDK. [#16465]

## Quantization

6. **conv2d MUST use per-channel quantization** — per-tensor causes severe accuracy loss due to weight variance across channels. [#15954]

7. **8a8w is NOT recommended for LLMs** — activations are too sparse for 8-bit. Use 16-bit activations (16a4w or 16a8w). [#15954, #16013]

8. **Calibration range depends on max_seq_len** — changing max_seq_len without recalibrating causes wrong outputs. Use `--tasks wikitext --limit 1`. [#16615]

9. **NumPy >= 2.0 breaks QNN compilation** — use Python 3.12 + numpy < 2.0 (e.g., 1.26.4). Python 3.13+ requires numpy 2.x which is incompatible with PyQnnManagerAdaptor's pybind11 layer. [#16557, #18795]

## Export & Compilation

10. **Use `examples/qualcomm/oss_scripts/llama/` for LLMs**, NOT `examples/models/llama/`. The latter produces oversized/broken .pte files. [#10226, #11100]

11. **QNN LLMs require `qnn_llama_runner`**, not `llama_main`. The standard runner is incompatible with QNN-exported models. [#11100]

12. **The .pte is compiled for a specific SoC** — running on a different SoC causes `Request feature arch with value X unsupported`. Match `-m` flag to target device. [#11100, #4973]

13. **HTP PD memory limit (~2GB per graph)** — if exceeded, increase `num_sharding` or reduce max_seq_len. Error: `graph requires estimated allocation of X KB, limit is Y KB`. [#15954, #17782]

## Runtime & Device Setup

14. **Both LD_LIBRARY_PATH and ADSP_LIBRARY_PATH must be set** on device. Missing ADSP_LIBRARY_PATH causes skel load failure (`DspTransport.openSession qnn_open failed`). [#1176, #1527]

15. **Push correct arch-specific libs** — e.g., `libQnnHtpV73Skel.so` for SM8550, `libQnnHtpV75Stub.so` for SM8650. Wrong libs cause silent failures or arch mismatch. [#1176, #16535]

## Build & Setup

16. **Add `-DSUPPORT_REGEX_LOOKAHEAD=ON`** to CMake flags when building runner for Qwen models — required for correct tokenization. [#11807]

17. **QNN backend supports custom models** — not limited to examples in `examples/qualcomm/`. Any model whose ops are QNN-supported can be exported. [#10966]

18. **SA8155 is not in `QcomChipset`** — QNN-HTP starts at V68 in the current schema. [#1176]

19. **QNN FP16 cannot represent Float.NEGATIVE_INFINITY in attention masks** — use `-255.0f` or `-65535.0f` instead. Custom inference code using `Float.NEGATIVE_INFINITY` for causal masks produces gibberish decode output. The `qnn_llama_runner` handles this internally. [#18812]

## Debugging Quick Reference

- `debug=True` in `generate_qnn_executorch_compiler_spec()` enables verbose QNN logs [#18410]
- `dump_context_from_pte()` extracts context binary for analysis with `qnn-context-binary-utility` [#17755]
- Runner output goes to `adb logcat | grep ExecuTorch`, not stdout [#11100]
- `[QNN Partitioner Op Support]: aten.X | False` means op falls back to CPU [#5199]
- `QnnBackend_validateOpConfig failed 3110` means op quantization config is incompatible with target arch [#12747]
- `Failed to create transport for device, error: 4000` → skel library loading failure, verify with `qnn-net-run` first [#16415]
- `KeyError: 'aten.alias_copy.default'` → using old export flow, switch to `examples/qualcomm/oss_scripts/llama/` [#10895]
