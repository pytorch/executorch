---
title: "Quantization Rules"
category: QUANTIZATION
---

# Quantization Rules

## Export Flow
- Always use PT2E quantization: `prepare_pt2e` -> calibrate -> `convert_pt2e` [#1141]
- Use `to_edge_transform_and_lower()`, never the older `to_backend()` API [#10297]

## LLM Quantization
- For QNN/Qualcomm LLMs: use `examples/qualcomm/oss_scripts/llama/`, NOT `examples/models/llama/export_llama.py --qnn` -- the latter has bugs producing oversized PTE files and gibberish output [#10226, #11034]
- Basic PTQ via `export_llama.py --pt2e_quantize qnn_*` produces poor accuracy for small LLMs (1B params). The improved flow in `oss_scripts/llama` is required [#11034]
- If a 4-bit quantized PTE is larger than the float model, you're hitting a known bug (fixed in PR #12167). Switch scripts [#10226]

## Scheme Selection
- 8a8w: general purpose, good accuracy-performance tradeoff [#10226]
- 16a4w: LLMs on Qualcomm HTP, maximizes weight compression [#10226]
- 16a8w: LLMs where 4-bit accuracy is insufficient

## XNNPACK Specifics
- Only float ops are delegated; integer ops fall through to CPU [#10297]
- `torch.mm` with two dynamic inputs is NOT delegated -- needs one constant weight [#10297]
- Enable `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` for better CPU fallback performance [#10297]
- BF16 NOT supported for delegation -- use fp16 instead. BF16 falls through to CPU [#10188]
- `batch_norm` only delegated after conv (fusion). Standalone batch_norm falls through [#1340]
- Never pass raw int8 tensors -- int8 is XNNPACK internal representation [#10960]
- Dynamic quantization still requires calibration before convert_pt2e [#11355]
- GRU quantization requires explicitly passing initial hidden state [#10104]

## dim_order Issues
- XNNPACKQuantizer + `to_edge_transform_and_lower` can produce `_to_dim_order_copy` ops that XNNPACK rejects: `RuntimeError: XNNPACK backend only supports contiguous memory format` [#11523]
- Workaround: disable dim order in EdgeCompileConfig [#10451]

## CoreML
- ET CoreML uses `torch.export.export` path, not `torch.jit.trace` [#10179]
- ct.ImageType is NOT supported; bake input normalization into the model wrapper [#10179]
- Accuracy drops from missing normalization can be severe (mIoU 0.57 -> 0.30) [#10179]
- Integer ReLU, integer mm, int16 ops are NOT supported [#11693]
- Tensors with rank > 5 are NOT supported [#11694]
- addmm with integer alpha/beta fails; cast to float [#11689]

## Calibration
- Use representative data matching inference distribution
- For LLMs: wikitext evaluation tasks work well for calibration [#10226]

## Debugging
- Use ETDump Inspector to check delegation rate: `inspector.print_data_tabular()` [#10297]
- Non-delegated `aten.mm.default` can consume 68%+ of inference time [#10297]
- Extract `.mlpackage` from PTE for direct CoreML comparison [#10179]

## Quantized Input/Output
- Possible to remove input/output q/dq nodes for integer-only pipelines [#1141]
- Use `FixedQParamsQuantizationSpec` for fixed-point formats (Q7, Q15) [#1141]
- Must preserve quantization params externally when removing q/dq nodes [#1141]
