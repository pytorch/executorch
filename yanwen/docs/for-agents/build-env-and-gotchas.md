# For coding agents: quant-dev coopmat build env, Plan A wiring, gotchas

Worktree: `/local/yanwen.xu/workspace/quant-dev/executorch`, branch `yanwen/quant-dev-local`
(tracks `origin/yanwen/quant-dev` = PR #19892, aggregated int4+int8 coopmat on merged fp16 #19009).
**All Plan A changes are dirty / uncommitted. Do not commit/push unless asked.**

## PTE storage location (MANDATE)
**All `.pte` files go to `/local/yanwen.xu/workspace/pte_out/`** — the single source of truth, no
duplicates elsewhere. Export scripts write there; phone deploy pushes from there.

## Paths (post doremy→yanwen migration; old /home/doremy/... is gone)
| Thing | Path |
|---|---|
| Model (Meta bf16) | `/local/yanwen.xu/models/llama3_1_8b/original/` (consolidated.00.pth, params.json, tokenizer.model) |
| Android NDK r29 | `/local/yanwen.xu/android-ndk-r29` |
| host glslc | `/sarc-c/gpusw/users/yanwen.xu/vulkan-sdk/1.4.341.1/x86_64/bin/glslc` |
| ccache / adb | `/usr/bin/ccache`, `/usr/bin/adb` |
| Built binaries | `/sarc-c/gpusw/users/yanwen.xu/artifacts/` |
| PTEs | `/local/yanwen.xu/workspace/pte_out/` |

## Env setup (uv + the editable requirement)
```bash
cd /local/yanwen.xu/workspace/quant-dev/executorch
uv venv .venv --seed && source .venv/bin/activate   # bash; user normally uses fish
./install_executorch.sh --minimal                   # first time; clones submodules
pip install -e . --no-build-isolation               # EDITABLE — required for op_registry/utils edits to take effect
```
`--minimal` alone is NON-editable (copies into site-packages); Python AOT edits then silently don't apply.
For the **C++ Android build you do NOT need editable** — cmake compiles worktree source directly.
`flatc` is at `.venv/bin/flatc`, only on PATH when the venv is **activated** (needed by `to_executorch()`).

## The two control knobs (Plan A)
- **`ET_VK_FORCE_BUFFER=1`** (EXPORT-time env): `VulkanPartitioner.__init__` injects
  `storage_type_override=BUFFER` → whole-graph buffer → coopmat-eligible PTE. Unset = texture (default ET).
  Read at partitioner construction, so works for both `export/export.py` CLI and custom scripts using
  `VulkanPartitioner({})`. An explicit `storage_type_override` in compile_options always wins.
- **`ET_VK_DISABLE_COOPMAT`** (RUNTIME env, set on the phone/binary): short-circuits the coopmat gates to
  the tiled fallback. Set = B-tiled baseline, unset = B-coopmat. No-op on a texture PTE (coopmat can't fire there).

## KEY finding: storage_type_override was DEAD before Plan A
`TagMemoryMetaPass` stored `self.default_storage` (from `storage_type_override`) but **never consumed it**;
`TensorRepSet.make_tensor_repr()` hard-coded "prefer texture". So the global override silently did nothing —
a stock PTE always picked texture (except tensors too big for texture, e.g. lm_head vocab=128256 → buffer).
Plan A A2 fix threads it through:
- `utils.py`: `make_tensor_repr(prefer_storage=TEXTURE_3D)` returns buffer when `prefer_storage==BUFFER` and
  `buffer_is_valid()`, else falls back to texture (an op lacking a buffer variant stays texture — **never crashes**).
- `utils.py`: `pick_representations(prefer_storage)` forwards it.
- `tag_memory_meta_pass.py`: passes `self.default_storage` into `pick_representations`.
Default arg stays `TEXTURE_3D`, so behavior is unchanged unless the override is set (87 repr tests still pass).

## How coopmat actually dispatches (so you don't chase the wrong shader)
- Runtime gates: `is_coopmat_eligible` (GemmCoopmat.h, fp16) and `can_use_q4gsw_coopmat`
  (QuantizedLinear.cpp, shared by all 3 quantized call sites: q4gsw, dq8ca_q4gsw, dq8ca_q8csw).
  Both now start with `if (std::getenv("ET_VK_DISABLE_COOPMAT")) return false;`.
- Gates require: `supports_cooperative_matrix()`, `subgroup_size()==64`, `storage_type_of(out)==kBuffer`,
  half dtype, M%64==0, N%64==0, K%32==0. fp16 gate ALSO has `!is_integrated_gpu()` (so fp16 coopmat is
  discrete-GPU only); the int4/int8 gate does NOT (fp16 won't fit the phone anyway).
- coopmat excludes gemv and needs M%64==0 → **fires only on PREFILL**. Decode (M=1) uses
  `linear_q4gsw_coop_*` (the gemv "coop" shader, different from coopmat WMMA). Forcing buffer makes decode
  use the `_coop_buffer_*` variants (they exist).
- In a built binary the fp16 coopmat shader is `linear_coopmat_*` (from coopmat_mm.yaml/glsl) — **grep
  `linear_coopmat`, not `coopmat_mm`**.

## Android build recipe (corrected paths)
```bash
export ANDROID_NDK_HOME=/local/yanwen.xu/android-ndk-r29
export ANDROID_NDK=$ANDROID_NDK_HOME
GLSLC=/sarc-c/gpusw/users/yanwen.xu/vulkan-sdk/1.4.341.1/x86_64/bin/glslc
cd /local/yanwen.xu/workspace/quant-dev/executorch && source .venv/bin/activate

# Step 1: core runtime + Vulkan backend
cmake . -Bcmake-out-android-vk --preset llm \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX=cmake-out-android-vk -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_PAL_DEFAULT=posix -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=OFF \
  -DGLSLC_PATH=$GLSLC \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-android-vk -j$(nproc) --target install --config Release

# Step 2: llama_main runner
cmake examples/models/llama -Bcmake-out-android-vk/examples/models/llama \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_INSTALL_PREFIX=cmake-out-android-vk -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_VULKAN=ON -DSUPPORT_REGEX_LOOKAHEAD=ON \
  -DPYTHON_EXECUTABLE=python -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-android-vk/examples/models/llama -j$(nproc) --config Release
# -> cmake-out-android-vk/examples/models/llama/llama_main
```
Gotchas: use `--preset llm` (NOT `linux`). The `EXECUTORCH_BUILD_VULKAN`/`SUPPORT_REGEX_LOOKAHEAD`
"not used by the project" warnings on step 2 are benign. `-include algorithm` works around a missing include.
**C++ gate changes (`ET_VK_DISABLE_COOPMAT`) require a rebuild** to take effect on the phone.

### Built unified binary (2026-06-02)
`/sarc-c/gpusw/users/yanwen.xu/artifacts/llama_main_coopmat_unified` (15.4 MB, md5 9e4249d9645eb4621c9d0f051f8e7319).
One `llama_main` that runs ALL coopmat paths (fp16 `linear_coopmat`, 4w `linear_q4gsw_coopmat`,
8da4w `linear_dq8ca_q4gsw_coopmat`, int8 `linear_dq8ca_q8csw_coopmat` — all verified embedded via `strings`)
with the `ET_VK_DISABLE_COOPMAT` runtime gate compiled in. Push it to the phone as `llama_main_coopmat`
(the name `bench_phone.sh` expects). `ET_VK_FORCE_BUFFER` is correctly NOT in the binary — it's a Python/AOT
partitioner env, not runtime.

## Verify
```bash
python yanwen/scripts/smoke_test_plan_a.py                                  # AOT wiring + global-buffer lower
python -m pytest backends/vulkan/test/test_vulkan_tensor_repr.py -q         # 87 pass (backward compat)
```

## Misc gotchas
- export_llm CLI OOMs for fp16 (fp32 upcast ~44.6 GB > 45 GB box) — use `yanwen/scripts/export_fp16.py`.
- `prompt_2k.txt` single-shot prefill can `VK_ERROR_DEVICE_LOST` on the phone — reboot first / use a short prompt.
- adb "no permissions" → `adb kill-server && adb start-server` (Samsung SM-F966U1).
- Long background jobs: use the harness background runner, not `nohup ... &` (gets orphaned/killed).
