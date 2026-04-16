# Voxtral TTS Progress Handoff

This file is the single-source handoff for the current `examples/models/voxtral_tts`
work. It is written so the work can be resumed on another machine without needing
the full prior chat history.

Last updated: 2026-04-16

## Goal

Primary goal:

- Reproduce `mistralai/Voxtral-4B-TTS-2603` in ExecuTorch.
- Support offline generation first, then streaming.
- Target CPU/portable and XNNPACK first.
- Final quality gate is Apple STT on a canonical prompt and voice.

Canonical acceptance contract used throughout this work:

- Text: `Hello, how are you today?`
- Voice: `neutral_female`
- Seed: `42`
- Sample rate: `24000`
- Frame rate: `12.5 Hz`
- Audio frame structure: `1 semantic + 36 acoustic = 37 codes`
- Success bar: generated WAV must transcribe back to the prompt with Apple STT

Important:

- Codec parity is necessary but not sufficient.
- A WAV that decodes correctly at the codec stage can still fail STT if the
  generator path is wrong.

## Model + Repo Locations Used

ExecuTorch repo:

- `/Users/younghan/executorch`

Voxtral reference C implementation used as oracle:

- `/Users/younghan/project/voxtral-tts.c`

Model assets used during this work:

- `/Users/younghan/models/Voxtral-4B-TTS-2603`

Expected model directory contents:

- `consolidated.safetensors`
- `params.json`
- `tekken.json`
- `voice_embedding/`

Model source:

- Hugging Face model: `mistralai/Voxtral-4B-TTS-2603`

## Current Implementation Surface

Main Voxtral TTS files in ExecuTorch:

- `examples/models/voxtral_tts/model.py`
  Eager model definition, checkpoint loading, LLM decoder, flow-matching head,
  codec decoder, audio-token embedding.
- `examples/models/voxtral_tts/export_voxtral_tts.py`
  Export CLI for `model.pte` and `codec_decoder.pte`.
- `examples/models/voxtral_tts/voxtral_tts_runner.cpp`
  C++ runner for offline and streaming generation.
- `examples/models/voxtral_tts/main.cpp`
  CLI entrypoint for the runner.
- `examples/models/voxtral_tts/parity.py`
  Shared prompt and trace helpers.
- `examples/models/voxtral_tts/verify_export_parity.py`
  Method-level parity harness for eager vs export vs runtime.
- `examples/models/voxtral_tts/compare_parity_traces.py`
  Trace comparator for eager vs runner traces.
- `examples/models/voxtral_tts/verify_codec_export.py`
  Codec-only parity validation.
- `examples/models/voxtral_tts/verify_xnnpack_transcript.py`
  Layered acceptance script with Apple STT hard gate.
- `examples/models/voxtral_tts/test_eager_e2e.py`
  Eager end-to-end oracle runner.
- `examples/models/voxtral_tts/voice.py`
  Voice asset loading helpers.

Main tests added or extended:

- `examples/models/voxtral_tts/test_export_cli.py`
- `examples/models/voxtral_tts/test_parity.py`
- `examples/models/voxtral_tts/test_validation_contract.py`
- `examples/models/voxtral_tts/test_verify_codec_export.py`
- `examples/models/voxtral_tts/test_verify_export_parity.py`

Current git note:

- `git status --short -- "examples/models/voxtral_tts"` reported the directory as
  untracked at the time this handoff was written. Treat this whole directory as
  in-progress local work, not landed repo state.

## What Has Been Implemented

The repo now contains a working Voxtral TTS implementation surface with:

- Eager FP32 model load from the original Mistral checkpoint.
- Prompt construction aligned to `mistral_common` speech request encoding.
- Voice embedding splice over `[AUDIO]` placeholder positions.
- Split export into:
  - `model.pte` for token embedding, text decoder, semantic head, predict velocity
  - `codec_decoder.pte` for codec decode
- C++ runner with:
  - offline mode
  - streaming mode
  - voice loading from `.pt` and `.bin`
  - trace JSON emission
  - seed control
- Method-level parity harness for:
  - `token_embedding`
  - `text_decoder`
  - `semantic_head`
  - `predict_velocity`
  - `audio_token_embedding`
- Layered acceptance script that:
  - exports
  - runs the C++ runner
  - validates codec separately
  - runs Apple STT
  - emits a manifest-style result bundle

## Major Changes Made During This Work

### 1. Decoder quantization scoping

Selective decoder quantization was added to isolate quality regressions:

- New CLI and helper parameter: `--decoder-qlinear-scope`
- Supported values:
  - `all`
  - `attention`
  - `feed_forward`
  - `none`

This was wired through:

- `export_voxtral_tts.py`
- `verify_export_parity.py`
- `verify_xnnpack_transcript.py`
- associated unit tests

Best quantized policy discovered so far:

- decoder `feed_forward`-only quantization is better than quantizing decoder
  attention or the whole decoder

Reason:

- it preserved semantic behavior better than the more aggressive alternatives

### 2. Better semantic diagnostics

`verify_export_parity.py` gained stronger semantic reporting:

- `semantic_triplet_report(...)`
- top-k semantic logit reporting
- explicit reporting on quantized seed-hidden semantic behavior

This made it easier to separate:

- hidden-state drift
- semantic drift
- runtime-only drift

### 3. Codec validation was separated from generator debugging

`verify_codec_export.py` was fixed to support:

- exact frame decode when possible
- padded decode to `max_codec_frames` when needed
- trim-to-valid-samples comparison

This was important because codec shape mismatches were previously polluting
generator debugging.

Known codec result from the last good validation path:

- codec validation passed with `max_abs_diff ~= 7.69e-07`

Conclusion:

- the main remaining bug is upstream of the codec

### 4. Eager oracle bug was found and fixed

Very important discovery:

- `test_eager_e2e.py` defined `_patch_eager_sdpa(model)` because
  `llama.custom_sdpa` may not behave correctly in eager CPU mode
- but the script did not actually call `_patch_eager_sdpa(model)`

This meant older eager WAVs were not reliable ground truth.

Patch applied:

- `test_eager_e2e.py` now calls `_patch_eager_sdpa(model)` immediately after
  `load_model(...)`
- KV caches are zeroed after patching

Impact:

- old eager failures must not be treated as authoritative architecture failures

## High-Confidence Findings

These are the facts I would trust most.

### 1. The checkpoint and voice assets are fine

Using the same model directory and same prompt with the C reference implementation
works.

Reference build:

```bash
cd /Users/younghan/project/voxtral-tts.c
make apple
```

Reference run:

```bash
./voxtral_tts \
  -d "/Users/younghan/models/Voxtral-4B-TTS-2603" \
  -v neutral_female \
  -s 42 \
  -o "/tmp/voxtral_tts_reference_hello.wav" \
  "Hello, how are you today?"
```

Observed reference result:

- generated `40` frames
- about `3.20s` audio
- Apple STT transcript: `Hello how are you today`

This is the strongest proof that:

- the downloaded Mistral checkpoint is valid
- the voice asset is valid
- the canonical prompt itself is valid

### 2. The quantized ExecuTorch runner still fails intelligibility

Best recent quantized candidate tried:

- XNNPACK
- `8da8w`
- decoder quantization scope `feed_forward`

Key run observation:

- increasing `--max_new_tokens` from `20` to `80` fixed an earlier truncation issue
- the runner then generated `44` frames
- it reached `END_AUDIO`
- output duration was about `3.52s`
- Apple STT still returned `No speech detected`

Conclusion:

- `max_new_tokens=20` was too small for this prompt
- but truncation was not the root cause of unintelligibility

### 3. The reference C path and ExecuTorch diverge before codec decode

Using the patched eager oracle vs the quantized runner:

- `prompt_token_ids` match
- `voice_len` matches
- `prefill_hidden` still diverges
- `frame0_hidden` diverges badly
- semantic behavior diverges by frame 1

Concrete trace comparison from the patched eager trace vs the runner trace:

- `prefill_hidden max_abs_diff ~= 0.4822`
- `frame0_hidden max_abs_diff ~= 9.5813`
- frame 0 semantic token still matches: `10`
- frame 1 semantic token diverges immediately:
  - eager: `10`
  - runner: `855`

This is the most important current localization:

- the bug is not "just codec"
- the split is already happening in or around the generator path before final decode

### 4. The eager patch improved the oracle substantially

Patched eager run:

```bash
python -u examples/models/voxtral_tts/test_eager_e2e.py \
  --model-path "/Users/younghan/models/Voxtral-4B-TTS-2603" \
  --text "Hello, how are you today?" \
  --output "/tmp/voxtral_eager_patched.wav" \
  --trace-json "/tmp/voxtral_eager_patched_trace.json" \
  --max-frames 60 \
  --seed 42
```

Observed result:

- generated `29` frames
- reached `END_AUDIO` at frame `29`
- waveform range looked healthy: about `[-0.3225, 0.3731]`
- Apple STT transcript was `No`

This is not correct yet, but it is much better than the earlier stale eager runs
that produced `No speech detected`.

Interpretation:

- the eager path is not yet perfect
- but older eager artifacts were definitely misleading

### 5. `custom_sdpa` alone is not the main explanation

I ran a direct A/B comparison:

- same Python model weights
- same prompt
- same voice
- same seed decode
- only difference: default `custom_sdpa` path vs patched eager fallback

Observed differences:

- `prefill_hidden max_abs ~= 1.55e-05`
- `seed_hidden max_abs ~= 0.001395`
- semantic top-5 and semantic argmax were the same

Conclusion:

- `custom_sdpa` vs eager fallback is a real difference
- but it is too small at prefill/seed to explain the full runner failure by itself

## Things That Were Misleading

These are the traps I would avoid repeating.

### 1. Old eager WAVs are not trustworthy

Do not use the earlier eager artifacts as architecture proof.

Why:

- `test_eager_e2e.py` was missing the call to `_patch_eager_sdpa(model)`

### 2. Post-frame-0 acoustic code comparisons across languages are noisy

Do not over-interpret C/Python/C++ acoustic code mismatches after frame 0 unless
the exact flow noise tensor is shared.

Reason:

- even with the same seed, the C reference, Python eager path, and C++ runner do
  not necessarily use the same RNG implementation
- once flow noise differs, acoustic codes diverge even if the semantic path is fine

Safe parity signals:

- prompt token IDs
- voice splice position and length
- prefill hidden
- seed hidden
- semantic logits
- frame 0 semantic token

Unsafe parity signal unless noise is shared:

- acoustic codes after the first branch through random flow noise

### 3. `max_new_tokens=20` is too low for the canonical prompt

This caused a false failure mode earlier.

Use a larger budget while debugging, for example:

- `60`
- `80`

## Current Best Understanding Of The Main Blocker

The remaining blocker is:

- generator path mismatch before codec decode

More specifically:

- prompt structure seems correct
- voice splice seems correct
- custom/eager decoder math is close at prefill/seed
- codec can be validated independently
- but the runner/export/runtime path still drifts enough before or during frame 0
  generation that final audio is unintelligible

Most likely remaining problem areas:

1. `text_decoder` export/runtime semantics
   - cache position handling
   - state reset across calls
   - method-level export/runtime behavior under XNNPACK

2. first-step generator orchestration in the runner
   - the transition from prompt prefill to seed decode to frame-0 generation

3. flow-matching parity at frame 0 under export/runtime
   - not because the ODE idea is wrong
   - but because the exported/runtime hidden state or per-step inputs are already off

## Known Good / Known Bad Snapshot

### Known good

- C reference implementation with the same checkpoint and same voice
- Apple STT exact match on the canonical prompt

### Known partially good

- patched eager Python path produces actual speech-like audio
- Apple STT hears `No`

### Known bad

- latest quantized ExecuTorch XNNPACK runner path still gives `No speech detected`

## Recommended Next Steps

If resuming on another machine, do the following in order.

### Step 1. Re-establish the external oracle first

Build and run the C reference again:

```bash
cd /path/to/voxtral-tts.c
make apple
./voxtral_tts -d "/path/to/Voxtral-4B-TTS-2603" -v neutral_female -s 42 \
  -o "/tmp/voxtral_tts_reference_hello.wav" "Hello, how are you today?"
swift /path/to/executorch/examples/models/voxtral_tts/transcribe_apple_speech.swift \
  "/tmp/voxtral_tts_reference_hello.wav" en-US
```

Do not continue unless this still transcribes correctly.

### Step 2. Use the patched eager script as the Python oracle

Run:

```bash
python -u examples/models/voxtral_tts/test_eager_e2e.py \
  --model-path "/path/to/Voxtral-4B-TTS-2603" \
  --text "Hello, how are you today?" \
  --output "/tmp/voxtral_eager_patched.wav" \
  --trace-json "/tmp/voxtral_eager_patched_trace.json" \
  --max-frames 60 \
  --seed 42
```

Do not use older eager artifacts.

### Step 3. Run plain FP32 export/runtime before quantization

This is the single highest-value next experiment.

Question to answer:

- Does FP32 XNNPACK export/runtime already fail STT?

If yes:

- the blocker is export/runtime semantics, not quantization

If no:

- quantization is the blocker, and the next work should stay inside the
  quantization boundary

### Step 4. Compare only stable parity signals first

When comparing traces, prioritize:

- `prompt_token_ids`
- `voice_len`
- `prefill_hidden`
- `seed_hidden`
- `frame0_hidden`
- semantic logits / semantic argmax

Do not spend too much time on acoustic code equality across implementations until
the exact same flow noise tensor can be injected everywhere.

### Step 5. Make flow noise injectable

Best next instrumentation improvement:

- allow the runner and parity harness to accept an explicit initial `x0` flow
  noise tensor for frame 0

That would remove the RNG confounder and make acoustic parity meaningful again.

### Step 6. Keep codec debugging separate

Do not reopen codec debugging unless generator parity regresses again.

Current evidence says:

- codec path is good enough
- generator path is the blocker

## Concrete File-Level TODOs

If I were continuing immediately, I would focus in this order:

1. `examples/models/voxtral_tts/test_eager_e2e.py`
   - keep using the patched eager fallback
   - validate whether STT can be improved from `No` toward the full phrase

2. `examples/models/voxtral_tts/export_voxtral_tts.py`
   - export plain FP32 XNNPACK artifacts and test them end-to-end

3. `examples/models/voxtral_tts/voxtral_tts_runner.cpp`
   - add even denser trace fields if needed:
     - `seed_hidden`
     - `frame0_audio_embed`
     - `frame1_hidden`
     - optional injected flow noise for frame 0

4. `examples/models/voxtral_tts/verify_export_parity.py`
   - keep method-level parity focused on hidden states and semantic behavior first
   - avoid over-weighting post-noise acoustic mismatches

5. `examples/models/voxtral_tts/verify_xnnpack_transcript.py`
   - note that the current default in the file is still:
     - `DEFAULT_ACCEPTANCE_QLINEAR = "8da4w"`
   - but the more promising candidate during debugging was:
     - `8da8w` with `decoder_qlinear_scope=feed_forward`
   - align the acceptance default only after FP32 behavior is understood

## Commands Worth Keeping

Build ExecuTorch runner:

```bash
cd /Users/younghan/executorch
make voxtral_tts-xnnpack
```

Run quantized ExecuTorch candidate:

```bash
cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
  --model "/tmp/.../model.pte" \
  --codec "/tmp/.../codec_decoder.pte" \
  --tokenizer "/Users/younghan/models/Voxtral-4B-TTS-2603/tekken.json" \
  --voice "/Users/younghan/models/Voxtral-4B-TTS-2603/voice_embedding/neutral_female.pt" \
  --text "Hello, how are you today?" \
  --output "/tmp/accepted.wav" \
  --trace_json "/tmp/runner_trace.json" \
  --max_new_tokens 80 \
  --seed 42
```

Run Apple STT:

```bash
swift examples/models/voxtral_tts/transcribe_apple_speech.swift \
  "/tmp/output.wav" en-US
```

Compare traces:

```bash
python examples/models/voxtral_tts/compare_parity_traces.py \
  --reference "/tmp/voxtral_eager_patched_trace.json" \
  --candidate "/tmp/runner_trace.json"
```

## Final Bottom Line

The work is no longer in the "unknown architecture" phase.

We now know:

- the original checkpoint works
- the C reference is a valid behavioral oracle
- codec validation is mostly solved
- the acceptance failure is not just truncation
- the main remaining problem is generator parity before codec decode
- old eager failures were partly caused by a broken eager oracle setup

The most important next experiment is:

- plain FP32 XNNPACK export -> runner -> Apple STT

That one result should decide whether the remaining effort belongs mostly in:

- export/runtime correctness

or

- quantization recovery
