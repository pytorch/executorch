# WhisperAttention for XNNPACK - Project Plan

## Overview
Enable `WhisperCrossAttention` (WhisperAttention) for whisper-small model with XNNPACK recipe in optimum-executorch.

**Status**: In Progress - Pivoting to custom_ops-only kernel integration (no intrusive portable_lib build hooks)

---

## Completed Work

### 1. Python Custom Ops Implementation ✅
**File**: `extension/llm/custom_ops/custom_ops.py`

Added `Library` API definitions for:
- `executorch::alias` - Returns two tensors unchanged (pass-through)
- `executorch::alias.out` - Out variant with mutable outputs
- `executorch::update_cross_attn_cache` - Updates cross-attention KV cache
- `executorch::update_cross_attn_cache.out` - Out variant

All implemented as `CompositeExplicitAutograd` kernels that decompose into primitive ops.

### 2. C++ Runtime Kernels ✅
**Files**:
- `extension/llm/custom_ops/executorch_alias.cpp`
- `extension/llm/custom_ops/executorch_update_cross_attn_cache.cpp`

Both use manual kernel registration (not `EXECUTORCH_LIBRARY` macro) due to multi-output constraints.

### 3. WhisperAttention Modifications ✅
**File**: `optimum-executorch/optimum/executorch/attentions/whisper_attention.py`

Modified `recompute_kv` to use decomposed linear (matmul + add) instead of `nn.Linear` to avoid XNNPACK delegation issues inside `torch.cond`.

### 4. Build Configuration ✅
- `shim_et/xplat/executorch/build/build_variables.bzl` - Added C++ source files to `custom_ops`
- `extension/llm/custom_ops/CMakeLists.txt` - Removed intrusive `portable_lib` linkage approach
- `kernels/prim_ops/register_prim_ops.cpp` - Reverted incorrect registrations

### 5. Export Success ✅
Model exported successfully: `whisper_small_whisperattention.pte` (1.13GB)

---

## Current Blocker

### Out-Variant Runtime Path Is Incomplete

**Symptom**:
```python
# Registered operators (from _get_operator_names())
'executorch::alias.out'                    # ❌ missing at runtime
'executorch::update_cross_attn_cache.out'  # ❌ missing at runtime
```

**Error when running text_decoder**:
```
[method.cpp:788] Missing operator: [3] executorch::alias.out
[operator_registry.cpp:256] kernel 'executorch::update_cross_attn_cache.out' not found
```

**Root Cause**:
- `to_executorch()` requires out variants for these ops.
- Runtime registration should come from the `custom_ops` shared/static libs, not invasive direct edits to `portable_lib` linkage.
- `executorch::alias.out` is multi-output, so `EXECUTORCH_LIBRARY` cannot register it directly (single mutable tensor limitation in `make_boxed_kernel` wrapper).
- `executorch::update_cross_attn_cache.out` currently uses a full-cache copy path; it needs an in-place fast path for performance.

**Direction**:
- Keep kernel integration in `extension/llm/custom_ops` only.
- Avoid special-case `portable_lib` build wiring.
- Ensure runtime loads custom ops library before `_load_for_executorch` execution path.

---

## Next Steps

### 1) Build-System Scope (Immediate)
- Keep only minimal source-list updates for `custom_ops` (already in `build_variables.bzl`).
- Do not add new `portable_lib` CMake/Bazel wiring for these kernels.

### 2) `executorch::alias` Plan
- Preferred: remove dependency on a dedicated alias runtime kernel by expressing aliasing via view semantics (`view_copy`/`et_view`) in model code where possible.
- If `executorch::alias` must remain, keep `executorch::alias.out` kernel in `custom_ops` and implement it with `et_view` behavior (no `memcpy`) so outputs alias inputs.
- Keep manual registration for alias (multi-output op), but only through `custom_ops` library loading.

### 3) `executorch::update_cross_attn_cache` Plan
- Keep out-variant schema and runtime kernel in `custom_ops`.
- Add in-place fast path for KV cache update:
  - update only `[ :, :, :S, : ]` region;
  - avoid full-cache clone/copy in the common path;
  - fallback copy path only if `out` and `cache` do not alias.
- Reuse patterns from `extension/llm/custom_ops/op_update_cache.cpp` for correctness checks and stride-aware copy.

### 4) Runtime Loading Contract
- Ensure the runtime path imports/loads `extension/llm/custom_ops` before `_load_for_executorch` so custom kernels register into the global op registry.
- Verify registration with `_get_operator_names()` after the custom ops module is loaded.

---

## Testing Plan

After implementing the above:

1. **Out-variant export check**:
   ```python
   edge = to_edge(torch.export.export(model, example_inputs))
   edge.to_executorch()  # should not fail missing out variants
   ```

2. **Runtime registration check (after custom ops import/load)**:
   ```python
   from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
   from executorch.extension.pybindings._portable_lib import _get_operator_names
   assert "executorch::alias.out" in _get_operator_names()
   assert "executorch::update_cross_attn_cache.out" in _get_operator_names()
   ```

3. **Run text_decoder**:
   ```python
   m = _load_for_executorch('whisper_small_whisperattention.pte')
   encoder_output = torch.randn(1, 1500, 768)
   result = m.run_method('text_decoder', [encoder_output])
   ```

4. **Build whisper-cpu runner**:
   ```bash
   rm -rf cmake-out && make whisper-cpu
   ```

5. **Full inference test**:
   ```bash
   ./cmake-out/examples/models/whisper/whisper_runner \
     --model_path whisper_small_whisperattention.pte \
     --processor_path whisper_preprocessor.pte \
     --audio_path test_audio.wav \
     --tokenizer_path tokenizer/
   ```

---

## Files Modified

| File | Change |
|------|--------|
| `extension/llm/custom_ops/custom_ops.py` | Added executorch Library API definitions |
| `extension/llm/custom_ops/executorch_alias.cpp` | New C++ kernel |
| `extension/llm/custom_ops/executorch_update_cross_attn_cache.cpp` | New C++ kernel |
| `shim_et/xplat/executorch/build/build_variables.bzl` | Added 2 source files |
| `extension/llm/custom_ops/CMakeLists.txt` | Removed intrusive portable_lib linkage edits |
| `kernels/prim_ops/register_prim_ops.cpp` | Reverted incorrect registrations |
| `optimum-executorch/.../whisper_attention.py` | Decomposed nn.linear |
| `optimum-executorch/.../integrations.py` | Removed CUDA restriction |

---

## Notes

- **EXECUTORCH_LIBRARY macro limitation**: current wrapper expects exactly one mutable tensor argument; multi-output alias needs manual registration.
- **Out variant requirement**: `to_executorch()` requires out variants for custom ops.
- **Performance target**: `update_cross_attn_cache` should update KV cache in place in runtime kernels.

---

*Last updated: 2026-02-08*
