---
title: "Model Export Rules"
category: EXPORT_PATTERN
---

# Model Export Rules

## API Selection
- Always use `to_edge_transform_and_lower()`, never the older `to_edge()` + `to_backend()` pipeline [#10297]
- The newer API applies graph transforms that can cut inference time nearly in half (27s -> 16s in one case) [#10297]

## LLM Export
- For QNN backend: use `examples/qualcomm/oss_scripts/llama/`, not `examples/models/llama/export_llama.py` [#10226]
- Two Llama codebases exist: generic (examples/models/llama) vs QNN-optimized (examples/qualcomm/oss_scripts/llama). The generic path has known QNN bugs [#10226]
- Pass BOS/EOS metadata: `--metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'` [#10226]
- Pre-built CPU PTE files available at huggingface.co/executorch-community [#11034]

## dim_order Issues
- Default dim_order (v0.6+) introduces `_to_dim_order_copy` ops not recognized by CoreML or XNNPACK [#10451, #11523]
- CoreML: crashes at runtime because scalars are wrapped differently at compile vs runtime [#10451]
- XNNPACK: `RuntimeError: XNNPACK backend only supports contiguous memory format` [#11523]
- Workaround: disable dim_order in EdgeCompileConfig [#10451]

## CoreML Specifics
- Uses `torch.export.export` path, NOT `torch.jit.trace` [#10179]
- Does NOT support `ct.ImageType` -- bake normalization into model wrapper [#10179]
- Missing normalization causes severe accuracy drops (not a backend bug) [#10179]
- Extract `.mlpackage` from PTE for debugging: see `docs/source/backends/coreml/coreml-overview.md#extracting-the-mlpackage` [#10179]

## Vision Models
- MobileNetV3 + CoreML: requires disabling dim_order [#10451]
- YOLO12 + XNNPACK quantization: known dim_order failure, active issue [#11523]

## Dynamic Shapes
- torch.export requires static shapes; control flow on tensor values fails [#10297]
- Use `--disable_dynamic_shape` when dynamic shapes aren't needed [#10226]
- For LLMs: consider separate PTE files for prefill (batch>1) and decode (batch=1) [#10297]
- Runtime inputs must match export shapes; mismatches cause "Attempted to resize a static tensor" [#1350]

## Export Tracing
- For models with complex Python logic: try `export(model, inputs, strict=False)` [#11128]
- C/C++ tokenizers (HuggingFace pipelines, Stable Diffusion) cannot be traced at all [#10065]
- QAT quantization NOT supported in automated export pipeline [#13099]

## Performance Checklist
- Release build: `cmake -DCMAKE_BUILD_TYPE=Release` [#10297]
- Thread count: `get_threadpool()->_unsafe_reset_threadpool(4)` [#10297]
- Optimized kernels: `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` [#10297]
- Check delegation rate via ETDump Inspector [#10297]
- Non-delegated `mm` ops with dynamic inputs are the #1 perf killer [#10297]

## Build/Install
- v0.6+: `pip install executorch` includes CoreML + XNNPACK export support [#10066]
- MPS backend still requires building from source [#10066]
- Always run `git submodule sync && git submodule update --init` after clone [#1004]
- CMake 4.0 breaks the build -- pin cmake < 4.0 [#10152]
- Don't have ET repo in PYTHONPATH when using pip-installed package (shadows paths) [#2910]
- For tokenizer submodules (abseil-cpp, re2): use `--init --recursive` [#10063]

## LLM Tokenizers
- Qwen/Gemma: need `-DSUPPORT_REGEX_LOOKAHEAD=ON` for Android builds [#10867]
- iOS: need PCRE2-based regex_lookahead lib from executorch_llm xcframework (SwiftPM 1.1+) [#16391]
- Convert HF safetensors to consolidated.pth via torchtune utilities [#3303]

## iOS/Xcode
- Kernel libraries require `--force_load` linker flag for static initialization [#11221]
- Manual registration API (`register_<lib>_kernels()`) is being developed to replace this [#11221]
- SwiftPM: if PTE loading fails with Error 32, add `-all_load` linker flag [#14809]

## File Size Sanity Check
- Llama 3.2 1B float: ~2.4 GB, 8a8w: ~1.2 GB, 16a4w: ~0.8-1.1 GB [#10226]
- If quantized PTE > float model, you're hitting a known bug [#10226]
