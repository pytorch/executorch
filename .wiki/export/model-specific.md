---
title: "Model-Specific Export Patterns"
category: EXPORT_PATTERN
backends: []
last_validated: 2026-04-05
source_issues: [10226, 10031, 10297, 11034, 10179, 11523, 10451, 3303, 2805, 10867, 16391, 14809, 14025, 15914]
---

# Model-Specific Export Patterns

## LLM Export (Llama, Phi, Qwen)

### Choosing the Right Export Script

For Llama models, there are multiple export paths. Choose based on your target backend:

| Backend | Script | Notes |
|---------|--------|-------|
| XNNPACK (CPU) | `examples/models/llama/export_llama.py` | Generic path |
| QNN (Qualcomm) | `examples/qualcomm/oss_scripts/llama/llama.py` | Actively developed, better quantization |
| CoreML (Apple) | `examples/apple/coreml/scripts/export.py` | Use `--model_name` flag |

**Critical:** For QNN backends, do NOT use `examples/models/llama/export_llama.py` with `--qnn` flag -- it has known bugs producing oversized PTE files and poor accuracy. Use `examples/qualcomm/oss_scripts/llama/` instead. [Source: #10226]

### LLM Export with KV Cache

Export with `--use_kv_cache` for autoregressive decoding. This separates the model into prefill and decode phases:

```bash
python -m examples.models.llama.export_llama \
  --checkpoint <path>/consolidated.00.pth \
  -p <path>/params.json \
  --use_kv_cache \
  --disable_dynamic_shape \
  -d fp32 \
  --output_name="llama.pte"
```
[Source: #10226]

### Metadata for Token Control

Pass BOS/EOS token IDs via metadata:
```bash
--metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
```
[Source: #10226]

### Custom Kernels for LLMs

When using custom kernels (e.g., custom linear), see the ExecuTorch custom kernel documentation (search for "custom kernel libraries" in the ExecuTorch docs) for YAML codegen and build instructions. The LLM getting-started page references `replace_linear_with_custom_linear` but lacks detailed steps. [Source: #10031]

### Pre-built PTE Files

CPU-only pre-built PTE files are available at [huggingface.co/executorch-community](https://huggingface.co/executorch-community). Backend-specific PTE files are planned but not yet available. [Source: #11034]

### Converting HuggingFace Safetensors to PTH

ExecuTorch llama export scripts expect `consolidated.00.pth` format. To convert HuggingFace safetensors checkpoints, use torchtune utility functions. [Source: #3303]

### Llama vocab_size in params.json

If `params.json` has `vocab_size: -1` or the field is missing, the export scripts must infer it from the tokenizer. This is common with llama2 checkpoints. [Source: #2805]

### Qwen3/Gemma Tokenizer Requirements

Qwen and Gemma models use tokenizers with regex lookahead patterns (e.g., `(?!\S))`). The default RE2 regex engine does NOT support lookahead. [Source: #10867, #16391]

- **Android:** Add `-DSUPPORT_REGEX_LOOKAHEAD=ON` to your cmake build command [Source: #10867]
- **iOS (SwiftPM):** Use the `executorch_llm` xcframework from SwiftPM 1.1+ which includes PCRE2-based regex_lookahead support [Source: #16391]

## Vision Model Export

### MobileNetV3

MobileNetV3 export with CoreML partitioner requires disabling dim order due to `_to_dim_order_copy` ops not being supported by CoreML:

```python
et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[CoreMLPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
).to_executorch()
```
[Source: #10451]

### YOLO12

YOLO12 quantized with XNNPACKQuantizer cannot be lowered to XNNPACK due to dim_order issues:
```
RuntimeError: XNNPACK backend only supports contiguous memory format for inputs.
Expecting dim_order: (0, 1, 2), but got (2, 0, 1) for placeholder node
```
This is an active known issue. [Source: #11523]

### Semantic Segmentation Models

When exporting vision models (e.g., BiSeNetv2) that use input normalization:
- ExecuTorch via CoreML does NOT support `ct.ImageType` with scale/bias
- You must bake normalization into the model wrapper
- Missing normalization causes severe accuracy drops (e.g., mIoU 0.57 -> 0.30) [Source: #10179]

```python
# WRONG: Scale/bias computed but not applied
class ModelWrapper(torch.nn.Module):
    def forward(self, x):
        return self.model(x)  # Missing normalization!

# CORRECT: Bake normalization into the wrapper
class ModelWrapper(torch.nn.Module):
    def forward(self, x):
        x = (x - self.mean) / self.std  # Apply normalization
        return self.model(x)
```
[Source: #10179]

## Custom Op Handling During Export

### Selective Build for Custom Ops

When building C++ applications with ExecuTorch, you can selectively include only the ops your model needs:

```cmake
gen_selected_ops(
  LIB_NAME "select_build_lib"
  ROOT_OPS "aten::add.out"
  INCLUDE_ALL_OPS "OFF"
)
generate_bindings_for_kernels(
  LIB_NAME "select_build_lib"
  FUNCTIONS_YAML ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
)
gen_operators_lib(
  LIB_NAME "select_build_lib"
  KERNEL_LIBS ${_kernel_lib}
  DEPS executorch
)
```
[Source: #10297]

### Optimized Kernel Library

For ops that fall through XNNPACK to CPU (e.g., `native_layer_norm`), enable the optimized operator library:

```cmake
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
```

Then link `optimized_native_cpu_ops_lib` to your application. [Source: #10297]

## Dynamic Shapes Limitations

`torch.export` requires static shapes by default. When models use dynamic control flow (if/else based on tensor shapes):
- TorchScript can handle this dynamically but `torch.export` cannot
- For LLMs, use separate models for prefill (batch>1) and decode (batch=1) [Source: #10297]
- Consider `--disable_dynamic_shape` flag when dynamic shapes aren't needed [Source: #10226]

## iOS/macOS Deployment

### pip install vs Build from Source

As of v0.6:
- `pip install executorch` includes CoreML and XNNPACK export support on macOS
- MPS backend still requires building from source
- For iOS demo apps using all three backends (CoreML + XNNPACK + MPS), build from source is still needed [Source: #10066]

### SwiftPM Integration

For CoreML + XNNPACK only apps, use pip + SwiftPM without cloning the repo. For MPS, clone and build from source. [Source: #10066]

### SwiftPM PTE Loading: Error 32 (NotFound)

If the SwiftPM binary distribution fails to load PTE methods with Error 32 while Python runtime works fine, add the `-all_load` linker flag to ensure all symbols (including statically-initialized kernel registrations) are linked. [Source: #14809]

## Audio Model Export

### Voxtral/Whisper Audio Preprocessing

Audio input must be 16kHz sampling rate. Using 48kHz audio causes dimension mismatch errors. [Source: #14025]

```bash
# Resample audio to 16kHz before passing to the model
ffmpeg -i audio.mp3 -f f32le -acodec pcm_f32le -ar 16000 audio_input.bin
```

### Whisper mm Not Delegated

In Whisper models, `mm` nodes may not be delegated because weight tensors come from preprocessing (not recognized as model parameters). This is a known limitation with dynamically-constructed weights. [Source: #15914]

## See Also

- [Export Common Pitfalls](common-pitfalls.md) — General torch.export errors
- [Quantization Recipes](../quantization/recipes.md) — Quantization before export
- [QNN Quantization Guide](../backends/qnn/quantization.md) — QNN-specific LLM export + quantization
- [XNNPACK Overview](../backends/xnnpack/overview.md) — XNNPACK delegation during export
