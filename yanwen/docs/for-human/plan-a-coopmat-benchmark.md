# Plan A: one binary + storage flag + coopmat toggle — coopmat vs baseline benchmark

**Branch:** `yanwen/quant-dev-local` (worktree `/local/yanwen.xu/workspace/quant-dev/executorch`), based on PR #19892 (aggregated int4 + int8 coopmat) on top of merged fp16 coopmat (#19009). Changes are **dirty / uncommitted** by design.

## The idea in one line

Storage (texture vs buffer) is baked into the PTE at **export time** and can't change at runtime; the coopmat-vs-tiled choice is decided at **runtime**. So:

> **2 PTEs (texture / buffer) × 1 binary × 1 runtime env (`ET_VK_DISABLE_COOPMAT`) = the 3 configs you want.**

| Config | Storage (export) | `ET_VK_DISABLE_COOPMAT` (run) | What it is |
|---|---|---|---|
| **T-tiled** | texture | n/a (coopmat physically can't fire) | default ExecuTorch baseline |
| **B-tiled** | buffer | `=1` | fair, same-storage baseline |
| **B-coopmat** | buffer | unset | your coopmat |

Why coopmat needs buffer: the WMMA shaders use `coopMatLoad/Store` on buffers; the runtime gate requires `storage_type_of(out)==kBuffer`. By default ExecuTorch prefers texture (faster for the baseline), so coopmat never fires on a stock PTE — you must force buffer.

## Report THREE numbers (they answer different questions)

```
(T-tiled  →  B-coopmat)  =  (T-tiled → B-tiled)  +  (B-tiled → B-coopmat)
    total e2e gain            storage penalty          kernel gain (the fair one)
```

- **B-coopmat vs B-tiled** = pure kernel win (same storage). This is your "my shader is X% faster" claim.
- **B-tiled vs T-tiled** = the cost of switching texture→buffer (explains the gap).
- **B-coopmat vs T-tiled** = the honest e2e question: does going-buffer-to-get-coopmat beat stock ExecuTorch?

**Caveat:** coopmat only fires on **prefill** (M%64==0, non-gemv). Decode (M=1) is always gemv, unaffected by the toggle. So measure with a long prompt where prefill dominates (e.g. 2k prefill); decode tok/s will be ~equal across configs.

## How to produce the two PTEs

All PTEs go to `/local/yanwen.xu/workspace/pte_out/`.

### int4 (4w, 8da4w) and int8 (8da8w) — via the export_llm CLI
Same command you already use, run twice with the storage env. `ET_VK_FORCE_BUFFER` is read by `VulkanPartitioner.__init__`, so no script/config edit is needed:

```bash
cd /local/yanwen.xu/workspace/quant-dev/executorch && source .venv/bin/activate

# texture PTE (default ET)
python export/export.py <your llama vulkan 4w config...>          # -> *_4w_texture.pte (rename to pte_out)

# buffer PTE (coopmat-capable + fair baseline)
ET_VK_FORCE_BUFFER=1 python export/export.py <same config...>     # -> *_4w_buffer.pte
```
(Quant recipe seen in your prior runs: torchao int4, `block_size=(1,128)` group-128 weight. 8da4w adds dynamic per-token int8 activation; 8da8w = dynamic act + per-channel int8 weight, the path PR #19892 newly added.)

### fp16 — via the custom script (CLI OOMs on this box)
```bash
python yanwen/scripts/export_fp16.py                 # -> pte_out/llama3_1_8b_fp16_texture.pte
ET_VK_FORCE_BUFFER=1 python yanwen/scripts/export_fp16.py   # -> ..._fp16_buffer.pte
```
fp16 full (16 GB) only runs on a **discrete GPU** (won't fit the phone's 11.4 GB). For a phone-side dispatch sanity check use a layer subset: `N_LAYERS=8 ET_VK_FORCE_BUFFER=1 python yanwen/scripts/export_fp16.py`.

## How to run the 3 configs

Rebuild the Android binary first (the C++ gate change must be compiled in — see `yanwen/docs/for-agents/`). Then:

```bash
yanwen/scripts/bench_phone.sh llama3_1_8b_4w_buffer.pte llama3_1_8b_4w_texture.pte \
    "The history of computing began" 96
```
It pushes both PTEs and runs B-coopmat / B-tiled / T-tiled, printing the `PyTorchObserver {"prefill_token_per_sec":...}` line for each. For a real 2k-prefill number, reboot the phone first (frees GPU memory; the 2k single-shot prefill can `VK_ERROR_DEVICE_LOST` otherwise) and point `--prompt` at a 2k token file.

## Validate before trusting numbers

```bash
python yanwen/scripts/smoke_test_plan_a.py   # AOT wiring + global-buffer lowering, no GPU needed
```
And after the first buffer export, run a short prompt on the phone to confirm the model loads + emits coherent text before benchmarking.

## What changed (5 files, all dirty)
- `GemmCoopmat.h`, `QuantizedLinear.cpp`: `ET_VK_DISABLE_COOPMAT` getenv short-circuit at the top of the fp16 and int4/int8 coopmat gates.
- `utils.py`, `_passes/tag_memory_meta_pass.py`: made `storage_type_override` actually work (it was silently ignored — see for-agents doc).
- `partitioner/vulkan_partitioner.py`: `ET_VK_FORCE_BUFFER=1` injects `storage_type_override=BUFFER`.
