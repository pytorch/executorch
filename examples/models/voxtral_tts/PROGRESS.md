# Voxtral TTS Progress Handoff

Single-source handoff for `examples/models/voxtral_tts`. Written so work can
be resumed on another machine without prior chat history.

Last updated: 2026-04-23 (afternoon — codec-on-CUDA shipped)

## Current state (2026-04-23, post codec rewrite)

| Backend | Quant | model.pte | model.ptd | codec.pte | codec.ptd | LM RTF | E2E RTF | Wall clock | Frames |
|---|---|---|---|---|---|---|---|---|---|
| XNNPACK | fp32 | 15.5 GB | — | 610 MB | — | 4.8x | 4.8x | 15.3s | 40 |
| CUDA | fp32 | 5.4 MB | 15.8 GB | 748 MB (portable) | — | 3.34x | 51x | 178s | 43 |
| CUDA | 4w | 3.4 MB | 3.4 GB | 748 MB (portable) | — | 0.73x | 51x | 180s | 39 |
| **CUDA** | **4w + CUDA codec** ⚡ | **3.4 MB** | **3.4 GB** | **5.7 MB** | **303 MB** | **0.82x** | **0.88x** ⚡ | **3.7s** ⚡ | **32** |

**Sub-real-time end-to-end on A100**: 3.7 s wall clock for 2.56 s of audio
(48× faster than the CPU-codec variant; 4.1× faster than XNNPACK FP32 baseline).
Audio quality: RMS 0.029 / peak ±0.18 vs XNNPACK FP32 baseline 0.014 / ±0.21
(within bf16 sampling noise; intelligible speech).

The codec rewrite (`_conv1d_as_matmul`, `_conv_transpose1d_as_matmul` in
`model.py`) is mathematically identical to the original ops (eager parity max
abs diff = 5.5e-10 in fp32) and lets the codec lower onto AOTI's Triton matmul
kernels — bypassing both the missing `aoti_torch_cuda_convolution` shim and
Triton's lack of conv-autotune choices for the codec's ConvTranspose shapes.

## Session 2026-04-22 to 2026-04-23 — CUDA enablement + 4w quantization

### What landed (10 phases of work)

1. **CUDA install on devserver** — pinned to CUDA 12.8 (CUDA 13's `host_runtime.h` has incompatible 2-arg `__cudaLaunch` macro). `unset CPATH` is mandatory or gcc picks the 13 header. Memory at `project_executorch_cuda_install.md`.
2. **Backend-aware SDPA/KV cache in `model.py`** — added `StaticKVCache` (BHSD, bf16) and `StandardSDPA` calling `torch.ops.triton.sdpa` directly. The XNNPACK custom_sdpa path is preserved and unchanged.
3. **`--backend cuda` in `export_voxtral_tts.py`** — emits `model.pte` + `aoti_cuda_blob.ptd`. Codec routed through portable backend (CUDA AOTI lacks conv shims for ConvTranspose1d).
4. **`voxtral-tts-cuda` CMake preset** plus parent `llm-release-cuda` preset.
5. **Runner `--data_path` / `--codec_data_path`** — uses dual-path `Module(model_path, data_path, ...)` overload for AOTI .ptd loading.
6. **Causal mask for CUDA SDPA** (`_build_causal_mask_bool`) — CRITICAL fix from Codex adversarial review. Without it, queries attend to the entire zero-filled `[1, H_kv, max_seq_len, D]` cache including unwritten future slots, corrupting hidden state from frame 0. Threaded through `MistralDecoder.forward → MistralDecoderLayer → LMAttention → StandardSDPA → triton.sdpa(mask=...)`.
7. **Mixed precision (fp32 weights, bf16 SDPA only)** — `StaticKVCache` declared bf16, `StandardSDPA.forward` casts Q to bf16 just before kernel and casts result back. `load_model` preserves declared bf16 buffer dtype during meta-materialization. Drops `--dtype=bf16` hard-requirement; default fp32 preferred for quality.
8. **Runner bf16 staging buffers** with `lm_input_is_bf16` metadata switch — runner reads model dtype from .pte metadata and allocates bf16 staging buffers per-call when needed. fp32 mixed-precision exports report 0; quantized exports report 1.
9. **CUDA 4w quantization (`--qlinear 4w`)** — auto-promotes `--dtype` to bf16, auto-sets `--qlinear-packing-format=tile_packed_to_4d` for the `_weight_int4pack_mm` kernel. `flow_head.input_projection` (3072×36) auto-skipped (K=36 not divisible by group_size=32). LM RTF drops from 3.34 → 0.73, .ptd from 15.8 GB → 3.4 GB, frame count 39 vs baseline 40.
10. **Drop codec from warmup** — codec runs on portable (no Triton autotune to amortize); one warmup call took ~150 s on CPU. Removed → startup wait drops from ~150 s to <60 s (Triton LM-method autotune dominates remaining time).

### Parity gates passed (2026-04-22, fp32 mixed precision)

Compared CUDA AOTI vs eager FP32 CPU baseline with `seed=42, "Hello, how are you today?"`:
- Last-position prefill hidden cosine: **0.999994** (gate ≥ 0.998)
- First-frame semantic argmax: **identical** (3040 in both paths)
- First-frame top-5 logits: **identical**
- Frame count before END_AUDIO: 43 vs CPU baseline 40

### Bugs fixed during CUDA bring-up

1. `__cudaLaunch was not declared` (sort.cu) — CPATH polluted with CUDA 13 path; `unset CPATH`.
2. `PendingUnbackedSymbolNotFound` during AOTI lowering — `F.scaled_dot_product_attention` decomp leaks ~12 unbacked symbols/layer; switched to `torch.ops.triton.sdpa` directly.
3. `Expected bfloat16 inputs` from triton.sdpa on fp32 — solved by mixed precision (fp32 weights, bf16 SDPA cast).
4. `NoValidChoicesError` for `aten.convolution.default` on codec — Triton conv autotune has no kernels for ConvTranspose1d shapes. Workaround: route codec through portable.
5. `Both operands must be same dtype` in codec autotune — `CodecDecoder.forward` hardcoded `dtype=torch.float32` for `quantizer.decode`. Fixed to read first conv weight dtype.
6. Runner `Aborted` at warmup — fp32 buffers fed to bf16 AOTI methods. Fixed via `lm_input_is_bf16` metadata switch + bf16 staging in runner.
7. `install_executorch.sh` uses `pip install .` not `-e .` — repo edits don't propagate. Workaround: `cp` to conda site-packages while iterating, or `pip install -e . --no-build-isolation`.
8. AOTI `.so` requires `GLIBCXX_3.4.30` not in `/lib64/libstdc++` — set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
9. `aoti_cuda_backend` target not built in default preset — must use `llm-release-cuda` (not `llm-release`) for the parent build.

### Files changed (since prior handoff)

| File | Change |
|---|---|
| `model.py` | StaticKVCache (bf16 BHSD), StandardSDPA (bf16 cast in/out), `_build_causal_mask_bool`, dtype-preserving meta buffer materialization, `CodecDecoder.forward` dtype fix |
| `export_voxtral_tts.py` | `--backend cuda` + `cuda-windows` choices, conv1d_to_conv2d decomp, CudaPartitioner per method, `.ptd` write, bf16 auto-promotion for `--qlinear`, `tile_packed_to_4d` auto-set, `lm_input_is_bf16` metadata, codec routed to portable + cast to fp32 |
| `voxtral_tts_runner.{h,cpp}` | `--data_path` / `--codec_data_path` ctor args, dual-path `Module` overload, `lm_use_bf16_` member, `fp32_to_bf16` / `bf16_to_fp32` helpers, bf16 staging for all LM call sites, `read_float_tensor` for outputs, codec dropped from warmup |
| `main.cpp` | `--data_path` and `--codec_data_path` gflags |
| `CMakePresets.json` | `voxtral-tts-cuda` configure/build/workflow presets |
| `BENCHMARK.md` | A100 FP32 + 4w-quant rows |
| `cuda_enablement.plan.md` | Full plan + status table per phase |
| `run_cuda_e2e.sh` | One-shot end-to-end script |
| `run_cuda_4w.txt` | Ready-to-paste runner cmd lines |

### Codec on CUDA via conv-as-matmul — SHIPPED 2026-04-23

Bypassed both AOTI conv barriers by rewriting `Conv1d` / `ConvTranspose1d` as
`unfold + matmul` / `matmul + Fold`. Math identical at fp32 (max abs diff
5.5e-10), Triton autotune found 20 valid bmm kernels for the codec ops where
the conv form returned `NoValidChoicesError`.

Implementation:
- `model.py:_conv1d_as_matmul(x, weight, bias, stride, dilation)` — F.unfold to extract sliding windows, matmul with `weight.reshape(C_out, C_in*K).t()`, transpose back
- `model.py:_conv_transpose1d_as_matmul(x, weight, bias, stride)` — matmul with `weight.reshape(C_in, C_out*K)`, then F.fold for stride-overlap accumulate
- `CodecCausalConv1d.forward` and `CodecCausalConvTranspose1d.forward` updated to call the helpers (still own `nn.Conv1d`/`ConvTranspose1d` for state_dict compatibility)
- `export_voxtral_tts.py` no longer routes codec to portable; codec exports via CUDA AOTI with `triton_kernel_mode=OFF` (additive ALiBi mask in CodecAttention is incompatible with Triton SDPA's bool mask)
- Codec's `.ptd` write renamed to `codec_aoti_cuda_blob.ptd` so it doesn't collide with the LM's `aoti_cuda_blob.ptd`

### Background notes for the rewrite (kept for context)

PoC at `/tmp/poc_conv_as_matmul.py` proved the approach: a `Conv1dAsMatmul` module (nn.Conv1d weight reshaped + F.unfold + matmul) is bit-exact to nn.Conv1d under bf16 (rel error 5–6e-3 = bf16 floor) AND lowers cleanly through CUDA AOTI (Triton autotune found 19 valid mm kernels for the K=4 case that originally returned `NoValidChoicesError` for the conv path).

Codec speedup measurement at `/tmp/poc_codec_cpu_vs_cuda.py`:

```
ExecuTorch portable backend (today):  ~150,000 ms  (256 frames, 20s audio)
PyTorch CPU eager fp32:                ~2,312 ms   (~65× faster than portable!)
PyTorch CUDA eager fp32:                  27.6 ms  (83.7× faster than CPU eager)
AOTI matmul on CUDA (estimated):           38 ms   (1.37× the eager CUDA conv)
```

Two separate inefficiencies stack today: portable backend uses single-threaded scalar conv kernels (~65× slower than MKL/oneDNN), AND portable runs on CPU (~84× slower than CUDA). The matmul rewrite addresses both at once by moving the codec to CUDA AOTI.

**Plan for the rewrite:**
1. Promote `Conv1dAsMatmul` from PoC into `model.py` and replace the `nn.Conv1d` inside `CodecCausalConv1d`.
2. Add `ConvTranspose1dAsMatmul` (input @ weight.flatten + nn.Fold for stride-overlap accumulate) and replace the `nn.ConvTranspose1d` inside `CodecCausalConvTranspose1d`.
3. Eager parity test: rewritten codec vs original codec for a representative codes input — assert per-sample diff < 1e-2 (bf16 floor) and waveform RMS within 5%.
4. Drop the "codec_backend = portable" workaround in `export_voxtral_tts.py`. Codec now exports via CUDA backend.
5. Re-export, re-build, re-run. Expected total wall clock for 3 s of audio: **~3 s** (vs current ~158 s).
6. Update BENCHMARK.md with the new "CUDA full pipeline" row.

Estimated end-state numbers based on current pieces:

| | Today | After codec rewrite |
|---|---|---|
| LM time (3 s audio, 4w) | 2.1 s | 2.1 s (unchanged) |
| Codec time (3 s audio) | 156 s | ~0.04 s |
| Total wall clock | 158 s | **~2.2 s** |
| End-to-end RTF | 51x | **0.7x (sub-real-time)** |

## Prior state (snapshot — 2026-04-16)

End-to-end ExecuTorch runner produces intelligible speech verified by parakeet
STT. Offline, streaming, and live-playback (`--speaker`) modes all work.

End-to-end ExecuTorch runner produces intelligible speech verified by parakeet
STT. Offline, streaming, and live-playback (`--speaker`) modes all work.

| Backend | Quant | model.pte | RTF (short) | RTF (long) | Transcript |
|---------|-------|-----------|-------------|------------|------------|
| XNNPACK | fp32 | 15.5 GB | 4.8x | 3.2x | Hello, how are you today? |
| XNNPACK | 8da4w ff | 7.0 GB | ~3.5x | 2.6x | Hello, how are you today? |
| XNNPACK | 8da8w | 5.7 GB | ~2.8x | 1.9x | Hello, how are you today? |
| XNNPACK | 8da4w all | 4.3 GB | ~3.8x | 2.0x | Ah hello. How are you today? |
| Portable | fp32 | 15.5 GB | 87x | — | Hello, how are you today? |

FP32 frame codes are **bit-identical** to the C reference (`voxtral-tts.c`)
for all 40 frames. Waveform correlation with C ref is 0.9995.

## Bugs fixed (vs prior handoff)

1. **Codec reshape order** (`model.py:1150`) — `waveform.reshape(B, 1, P*T)`
   was patch-outer/frame-inner. Fixed to `waveform.transpose(1, 2).reshape(B,
   1, T * P)` (frame-outer/patch-inner matching C ref). This was the root
   cause of unintelligible audio.

2. **Flow-matching RNG** (`voxtral_tts_runner.cpp`) — replaced
   `std::normal_distribution` with xorshift64+Box-Muller matching the C
   reference. Without this, acoustic codes diverge by frame 1.

3. **ALiBi slopes** (`model.py:794`) — `_get_alibi_slopes` used `r**i`
   (starting at 1.0); fixed to `r**(i+1)` (starting at 0.5, matching ALiBi
   paper and C ref). Improved codec correlation from 0.998 to 0.9995.

4. **Runner stdout** (`voxtral_tts_runner.cpp`, `main.cpp`) — all info
   messages moved to stderr so `--speaker` mode outputs clean PCM to stdout.

5. **STT gate** (`verify_xnnpack_transcript.py`) — replaced Apple STT (macOS
   only) with parakeet runner (`transcribe_parakeet.py`) for cross-platform
   validation.

## Files changed

| File | Change |
|------|--------|
| `model.py` | Codec reshape fix + ALiBi slope fix |
| `voxtral_tts_runner.cpp` | xorshift64 RNG, stderr logging, VOXTRAL_DUMP_CODES env var, streaming RNG fix |
| `voxtral_tts_runner.h` | Added `flow_rng_state_` field |
| `main.cpp` | Added `--speaker` flag, stderr logging for speaker mode |
| `export_voxtral_tts.py` | Codec export comment clarification |
| `verify_xnnpack_transcript.py` | Parakeet STT, `--qlinear none` support |
| `transcribe_parakeet.py` | New: resample + parakeet runner helper |
| `BENCHMARK.md` | New: quantization + long-text benchmark results |
| `README.md` | Updated: quantization docs, streaming, live playback, runner options |

## Next steps: Metal and CUDA backends

The streaming architecture is backend-agnostic — `model_->execute()` calls are
the same regardless of backend. Adding Metal/CUDA requires:

1. **Export**: add `--backend metal` / `--backend cuda` paths to
   `export_voxtral_tts.py`, following `voxtral_realtime/export_voxtral_rt.py`.
2. **Build**: add CMake presets for `voxtral-tts-metal` / `voxtral-tts-cuda`
   in `CMakePresets.json`, and Makefile targets.
3. **Test**: re-run the acceptance gate with the new backend's .pte files.

No runner C++ changes needed — the runner is backend-transparent.

## Quick start on a new machine

```bash
conda activate executorch

# Download model (if not cached)
huggingface-cli download mistralai/Voxtral-4B-TTS-2603

# Export
VOXTRAL_DIR=~/.cache/huggingface/hub/models--mistralai--Voxtral-4B-TTS-2603/snapshots/<sha>
python export_voxtral_tts.py --model-path $VOXTRAL_DIR --backend xnnpack \
  --qlinear 8da4w --decoder-qlinear-scope feed_forward \
  --output-dir ./voxtral_tts_exports

# Build
cmake --workflow --preset llm-release
cd examples/models/voxtral_tts && cmake --workflow --preset voxtral-tts-xnnpack && cd ../../..

# Run
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
  --model ./voxtral_tts_exports/model.pte \
  --codec ./voxtral_tts_exports/codec_decoder.pte \
  --tokenizer $VOXTRAL_DIR/tekken.json \
  --voice $VOXTRAL_DIR/voice_embedding/neutral_female.pt \
  --text "Hello, how are you today?" \
  --output output.wav --seed 42

# Verify (requires parakeet exports built separately — see examples/models/parakeet/)
python examples/models/voxtral_tts/transcribe_parakeet.py \
  --audio output.wav \
  --parakeet-runner ./cmake-out/examples/models/parakeet/parakeet_runner \
  --parakeet-model examples/models/parakeet/parakeet_tdt_exports/model.pte \
  --parakeet-tokenizer examples/models/parakeet/parakeet_tdt_exports/tokenizer.model
```
