# Gemma3 Metal Backend - Development Context

Tracking trials, failures, and fixes for running Gemma3 on the Metal backend.

## Goal

```bash
# Export
optimum-cli export executorch \
  --model "google/gemma-3-4b-it" \
  --task "multimodal-text-to-text" \
  --recipe "metal" \
  --dtype bfloat16 \
  --device mps \
  --output_dir="gemma3/gemma3-4b-it-metal"

# Build
make gemma3-metal

# Run
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path gemma3/gemma3-4b-it-metal/model.pte \
  --tokenizer_path /Users/younghan/project/executorch-examples/llm/apple/models/gemma3_tokenizer.json \
  --image_path docs/source/_static/img/et-logo.png \
  --temperature 0
```

## Architecture

- Gemma3-4B: head_dim=256, n_heads=8, n_kv_heads=4
- Metal SDPA kernel (`op_sdpa.mm`): only supports head_dim in {64, 96, 128}
- Metal backend uses AOTI (Ahead-Of-Time Inductor) compilation

## Changes Made

### 1. `make gemma3-metal` build target (DONE)

Files changed:
- `examples/models/gemma3/CMakeLists.txt` -- added Metal backend linking block
- `examples/models/gemma3/CMakePresets.json` -- added gemma3-metal configure/build/workflow presets
- `Makefile` -- added gemma3-metal target, .PHONY, help text

### 2. CMake 4.0 abseil C++17 detection fix (DONE)

**Problem**: CMake 4.0 deprecated `CMP0067`, so `CMAKE_CXX_STANDARD` was not
propagated to abseil's `check_cxx_source_compiles()`. Apple Clang defaults to
C++14, causing the check to fail. This set `ABSL_OPTION_USE_STD_STRING_VIEW=0`,
making `absl::string_view` a class that conflicts with sentencepiece's
`namespace absl { using std::string_view; }`.

**Fix**: Pre-set `ABSL_INTERNAL_AT_LEAST_CXX17 ON` as a cache variable in
`CMakeLists.txt` before the tokenizers subdirectory is added.

File changed: `CMakeLists.txt` (root)

### 3. Metal SDPA decomposition in optimum-executorch (DONE - output quality TBD)

**Problem**: Metal SDPA kernel only supports head_dim in {64, 96, 128}.
Gemma3 uses head_dim=256, causing runtime crash:
```
Error: aoti_torch_mps__scaled_dot_product_attention_math_for_mps(...) API call failed
```

**Fix**: Decompose SDPA into matmul + softmax in the Metal recipe before lowering,
following the pattern from `examples/models/voxtral_realtime/export_voxtral_rt.py`.

File changed: `optimum-executorch/optimum/exporters/executorch/recipes/metal.py`
- Added `_sdpa_decomposition()` -- decomposes SDPA into matmul + softmax
- Added `_linear_bias_decomposition()` -- decomposes linear+bias into matmul + add
- Applied decompositions via `ep.run_decompositions()` before `to_edge_transform_and_lower()`
- Conditional on `head_dim not in {64, 96, 128}`

### 4. torchao import chain fix in optimum-executorch (DONE)

**Problem**: `torchao.quantization.pt2e.quantize_pt2e` references
`torch.ao.quantization.quantizer` which doesn't exist in torch 2.11.0.dev20260215.
This is triggered by two import chains in the multimodal export path:

1. `_register_custom_attention()` unconditionally called
   `get_custom_sdpa_for_ring_kv_cache()` which imports from
   `executorch.examples.models.llama` -> `executorch.extension.llm.export.builder`
   -> `torchao`

2. `RemoveRedundantTransposes` import from `executorch.extension.llm.export.export_passes`
   triggers the same chain via `executorch.extension.llm.export.__init__`

**Fix**: Guard both imports behind `self.use_custom_sdpa` checks, since they are
only needed for custom SDPA (not standard SDPA used by Metal).

Files changed: `optimum-executorch/optimum/exporters/executorch/integrations.py`
- `_register_custom_attention()`: moved `get_custom_sdpa_for_ring_kv_cache()` call
  inside `if self.use_custom_sdpa:` guard (matching CausalLM version)
- `export()`: wrapped `RemoveRedundantTransposes` import/usage in
  `if self.use_custom_sdpa:` guard

File changed: `optimum-executorch/optimum/exporters/executorch/recipes/metal.py`
- Added `model.use_custom_sdpa = False` and `model.use_custom_kv_cache = False`
  before `model.export()` as safety net

## Failures Log

### Attempt 1: Initial `make gemma3-metal`
- **Error**: `absl::string_view` ambiguity during core lib build
- **Root cause**: Stale cmake cache + CMake 4.0 CMP0067 deprecation
- **Status**: FIXED (see change #2)

### Attempt 2: Runner with wrong model path
- **Command**: `--model_path gemma3/gemma3-4b-it-metal.pte`
- **Error**: Runner hung after tokenizer load (file not found, no error message)
- **Fix**: Correct path is `gemma3/gemma3-4b-it-metal/model.pte`
- **Status**: FIXED (user error)

### Attempt 3: Runner SDPA crash
- **Error**: `aoti_torch_mps__scaled_dot_product_attention_math_for_mps() API call failed`
- **Root cause**: Metal SDPA kernel doesn't support head_dim=256
- **Status**: FIXED in optimum-executorch recipe (see change #3), needs re-export

### Attempt 4: Re-export with `optimum-cli` - custom_sdpa import crash
- **Error**: `AttributeError: module 'torch.ao.quantization' has no attribute 'quantizer'`
- **Root cause**: Unconditional import of custom SDPA code triggers torchao incompatibility
- **Status**: FIXED (see change #4), needs retry

### Attempt 5: Re-export after fix #4 - RemoveRedundantTransposes import crash
- **Error**: Same `torch.ao.quantization.quantizer` AttributeError, different import site
- **Root cause**: `RemoveRedundantTransposes` import triggers same torchao chain
- **Status**: FIXED (see change #4 second bullet), needs retry

### Attempt 6: Runner libomp not found
- **Error**: `dyld: Library not loaded: /opt/llvm-openmp/lib/libomp.dylib`
- **Root cause**: Binary linked against OpenMP at `/opt/llvm-openmp/lib/` which doesn't exist.
  Homebrew libomp is at `/opt/homebrew/Cellar/libomp/21.1.5/lib/libomp.dylib`.
- **Fix**: `sudo mkdir -p /opt/llvm-openmp/lib && sudo ln -sf /opt/homebrew/Cellar/libomp/21.1.5/lib/libomp.dylib /opt/llvm-openmp/lib/libomp.dylib`
- **Status**: Needs user to run sudo command

### Attempt 7: Runner produces garbage output
- **Command**: Full pipeline (export + build + run) completed successfully
- **Output**: Model loads, tokenizes, and generates 99 tokens, but output is gibberish
  (random unicode from various scripts). PyTorchObserver stats show 271 prompt tokens,
  99 generated tokens, ~14.5s total inference.
- **Root cause**: Likely numerical issues in the SDPA decomposition (matmul + softmax
  path may accumulate precision errors differently than fused kernel), or the exported
  model has issues with bfloat16 on Metal. Needs further investigation.
- **Status**: OPEN -- pipeline works end-to-end, output quality needs fixing

## Environment

- macOS (Darwin arm64)
- CMake 4.0.3
- Apple Clang 17.0.0
- torch 2.11.0.dev20260215
- Conda env: `executorch`
- optimum-executorch: editable install from `/Users/younghan/project/executorch/optimum-executorch`
