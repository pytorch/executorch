# Qwen3-TTS Remediation Handoff

This file is a handoff note for continuing the qwen3-tts remediation work from a different agent.

Important:

- This file lives in the main workspace for discoverability.
- The actual in-progress code changes do **not** live in this checkout.
- The active remediation changes are in the worktree:

```text
/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation
```

- The worktree branch is:

```text
qwen3-tts-red-team-remediation
```

- The plan file already exists and should **not** be edited:

```text
/Users/younghan/.cursor/plans/qwen3-tts_remediation_edc4c5f6.plan.md
```

## 1. Resume Here

If another agent continues this work, start in the remediation worktree:

```bash
cd "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation"
git status --short --branch
```

Do **not** continue the code work in the main checkout at:

```text
/Users/younghan/project/executorch
```

That main checkout is only where this handoff file was written.

## 2. Current Todo Status

Plan todo state at the time this file was written:

- `repro-import-boundaries`: completed
- `decoder-clone-parity`: completed
- `runtime-hardening`: in progress
- `talker-export-validity`: pending
- `talker-end-to-end`: pending
- `streaming-cleanup`: pending
- `tests-and-docs`: pending

Important nuance:

- `decoder-clone-parity` was marked completed in the todo tracker because the code changes, unit tests, and runner build were verified.
- However, the original plan still called for a stronger upstream parity harness against real `Qwen3TTSModel.generate_voice_clone()` behavior. That full end-to-end harness has **not** been added yet.
- If strict plan fidelity matters more than the current todo state, consider reopening that verification gap later under `decoder-clone-parity` or folding it into `tests-and-docs`.

## 3. What Is Already Implemented

### 3.1 Reproducibility And Import Boundaries

Implemented in the remediation worktree:

- Created `examples/models/qwen3-tts/codec_io.py`.
- Created `examples/models/qwen3-tts/runtime_env.py`.
- Moved codec binary read/write helpers out of `model.py` into `codec_io.py`.
- Made `generate_codes.py` lazy-import `qwen_tts` only inside `main()`.
- Made `model.py` lazy-import `qwen_tts` decoder internals only inside `load_decoder_from_metadata()`.
- Made `export_qwen3_tts.py` lazy-import ExecuTorch lowering pieces only inside `lower_to_executorch()`.
- Added explicit environment preflight logic with:
  - validated `qwen-tts` version
  - validated `transformers` version
  - optional SoX check
- Added an explicit BF16 gate so `export_qwen3_tts.py --dtype bf16` fails early with an actionable error instead of emitting a known-bad path.
- Updated `README.md` to document:
  - the validated Python environment matrix
  - that BF16 is intentionally blocked for now

New tests added:

- `examples/models/qwen3-tts/tests/test_startup_and_env.py`

### 3.2 Decoder Semantics And Clone Parity

Implemented in the remediation worktree:

- `Qwen3TTSSpeechDecoderExport.forward()` now calls `decoder.chunked_decode(...)` instead of direct `decoder(...)`.
- `generate_codes.py` now has:
  - `--allow-silence-bootstrap`
  - `_validate_prompt_mode()`
  - `_prepare_codes_for_decode()`
  - `_build_codes_metadata()`
  - `_metadata_output_paths()`
- Clone-mode helper output now preserves prefix/reference codec context by:
  - prepending `prompt_dict["ref_code"]` to generated codes when present
  - writing `prefix_codes_len` into metadata
  - always writing the runner-visible sibling metadata sidecar at `codes_path.with_suffix(".json")`
- `main.cpp` now:
  - rejects text-only helper invocation unless either `--ref_audio` is provided or `--allow_silence_bootstrap` is explicitly passed
  - forwards `allow_silence_bootstrap` to the helper
- `qwen3_tts_runner.h/.cpp` now:
  - reads sibling metadata sidecar JSON for `prefix_codes_len`
  - trims decoded waveform proportionally after vocoder execution
  - clears waveform instead of erroring when `prefix_codes_len == codes_len`

New tests added:

- `examples/models/qwen3-tts/tests/test_decoder_clone_parity.py`

## 4. Files Currently Modified In The Remediation Worktree

At the time this note was written, `git status --short --branch` in the remediation worktree showed:

```text
## qwen3-tts-red-team-remediation
 M examples/models/qwen3-tts/README.md
 M examples/models/qwen3-tts/export_qwen3_tts.py
 M examples/models/qwen3-tts/generate_codes.py
 M examples/models/qwen3-tts/main.cpp
 M examples/models/qwen3-tts/model.py
 M examples/models/qwen3-tts/qwen3_tts_runner.cpp
 M examples/models/qwen3-tts/qwen3_tts_runner.h
?? examples/models/qwen3-tts/codec_io.py
?? examples/models/qwen3-tts/runtime_env.py
?? examples/models/qwen3-tts/tests/test_decoder_clone_parity.py
?? examples/models/qwen3-tts/tests/test_startup_and_env.py
```

These are still uncommitted.

No runtime-hardening, talker-export-validity, talker-end-to-end, streaming-cleanup, or docs/progress-file changes have been made yet beyond the files listed above.

## 5. Fresh Verification Evidence

### 5.1 Python Tests

The last full targeted Python regression run from the remediation worktree was:

```bash
conda run -n executorch python -m pytest \
  examples/models/qwen3-tts/tests/test_convert_weights.py \
  examples/models/qwen3-tts/tests/test_startup_and_env.py \
  examples/models/qwen3-tts/tests/test_decoder_clone_parity.py \
  -q
```

Result:

```text
13 passed in 5.62s
```

### 5.2 Python Entry Point Startup

These were verified successfully:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_qwen3_tts.py --help
conda run -n executorch python examples/models/qwen3-tts/generate_codes.py --help
```

The current machine environment intentionally still fails the real helper preflight in a friendly way because it is **not** yet in the documented supported state:

```bash
conda run -n executorch python examples/models/qwen3-tts/generate_codes.py \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text hello \
  --output-codes /tmp/qwen3_tts_test_codes.bin
```

Observed expected failure reason:

- `transformers==5.3.0` is installed instead of `4.57.3`
- `sox` is missing from `PATH`

This is expected with the new preflight code.

### 5.3 Runner Build

The qwen3-tts runner build was freshly re-verified in this session, but **not** from the raw remediation worktree path.

Important build constraint:

- ExecuTorch top-level CMake currently hard-fails unless the source directory name is exactly `executorch`.
- Therefore, direct `make qwen3-tts-cpu` from:

```text
/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation
```

will fail with the repo-name restriction.

The working build workaround is to create a symlink alias whose final path component is `executorch`.

Correct command:

```bash
cd "/Users/younghan/project/executorch"
ln -s "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation" ".worktrees/executorch"
```

Important:

- Use the absolute target path exactly as above.
- A previous relative symlink attempt was wrong and caused `cd` failures.

Then run:

```bash
git -C "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation" submodule update --init --recursive

cmake -S "/Users/younghan/project/executorch/.worktrees/executorch" --preset llm-release

cmake --build "/Users/younghan/project/executorch/.worktrees/executorch/cmake-out" --target install --parallel 8

cmake --preset qwen3-tts-cpu -S "/Users/younghan/project/executorch/.worktrees/executorch/examples/models/qwen3-tts"

cmake --build "/Users/younghan/project/executorch/.worktrees/executorch/cmake-out/examples/models/qwen3-tts" --target qwen3_tts_runner --parallel 8
```

Fresh result from this session:

- ExecuTorch configure via symlink alias succeeded
- ExecuTorch install build succeeded
- qwen3-tts runner configure succeeded
- qwen3-tts runner build succeeded

Last meaningful lines:

```text
[ 33%] Linking CXX executable qwen3_tts_runner
[100%] Built target qwen3_tts_runner
```

Warnings observed during the qwen3-tts-specific configure stage that did **not** block the build:

- duplicate-library warning during link
- optional library warnings such as:
  - `aoti_cuda_backend library is not found`
  - `flatccrt library is not found`
  - `etdump library is not found`
  - `bundled_program library is not found`
  - `metal_backend library is not found`
  - others in the same pattern

These warnings did not prevent `qwen3_tts_runner` from building in the verified path above.

## 6. Environment And Local State Gotchas

### 6.1 Helper Environment

The current `executorch` conda env is **not yet** suitable for real helper/export execution because:

- `qwen-tts==0.1.1` is expected
- `transformers==4.57.3` is required by the current preflight
- the machine currently has `transformers==5.3.0`
- `sox` is missing from `PATH`

This means:

- startup and help commands are now fixed and work
- real `qwen_tts` runs are intentionally blocked by preflight until the environment is corrected

If the next agent needs real generation/export instead of just unit tests:

1. install the supported `transformers` version
2. ensure `sox` is installed and on `PATH`
3. re-run the relevant helper/export commands

### 6.2 Worktree Move Limitation

Do **not** try to rename the remediation worktree using `git worktree move` after submodules are checked out.

Observed error:

```text
fatal: working trees containing submodules cannot be moved or removed
```

That is why the symlink alias workaround is used instead of renaming the worktree.

### 6.3 Symlink Alias Is Local Only

The `.worktrees/executorch` symlink is only a local build convenience artifact.

- It is not a committed repo change.
- If it goes missing, recreate it with the exact absolute symlink command shown above.

## 7. What Is Not Done Yet

### 7.1 Runtime Hardening

Status: in progress in the todo tracker, but no substantive code changes have been made yet for this workstream.

Still outstanding:

- Replace ad-hoc string scanning of `export_manifest.json` in `qwen3_tts_runner.cpp` with `nlohmann/json`.
- Replace ad-hoc metadata parsing in `read_codes_metadata()` with `nlohmann/json`.
- Harden codec file parsing:
  - header size validation
  - multiplication overflow checks
  - payload size checks
  - `codes_len * num_quantizers` consistency
  - `num_quantizers == exported metadata`
  - `0 <= code < codebook_size`
- Replace fixed temp file path in `main.cpp`:

```text
qwen3_tts_codegen_codes.bin
```

with a unique temp file and cleanup flow.
- Stop eager-loading all bucket models in `from_model_dir()`.
- Add bucket-load-aware latency accounting.
- Add negative tests for malformed codec input and malformed manifest input.

### 7.2 Talker Export Validity

Status: untouched so far.

Files not yet updated:

- `examples/models/qwen3-tts/export_talker.py`
- `examples/models/qwen3-tts/convert_talker_weights.py`
- `examples/models/qwen3-tts/config/talker_config.json`
- `examples/models/qwen3-tts/config/code_predictor_config.json`

Known outstanding issues from the plan:

- `export_talker.py` still uses `strict=False` for loading state dicts.
- warning-only invalid exports are still possible
- no strict config-vs-checkpoint validation exists yet
- no reusable talker manifest separating backbone vs aux weights exists yet

### 7.3 End-To-End ExecuTorch Talker Orchestration

Status: not started.

Missing:

- `examples/models/qwen3-tts/talker_exec.py`
- prefill/decode-step orchestration
- aux weight integration
- greedy parity harness vs upstream Qwen/HF

### 7.4 Streaming Cleanup

Status: not started.

`examples/models/qwen3-tts/streaming_generate.py` is still in the red-team state:

- requires `--talker-dir` even for decode-only path
- eagerly loads all decoder buckets
- accepts oversize `chunk-size` without proper rejection path
- reports chunk-concatenation as if it were true streaming
- latency accounting is not yet separated into honest metrics

### 7.5 Docs / Progress / Landing

Status: not started beyond the initial README env note.

Still needed:

- update `README.md` for streaming caveats and current supported flows
- update `PROGRESS.md` to reflect completed remediation steps
- add broader regression coverage
- possibly reopen or supplement `decoder-clone-parity` verification with true upstream harnessing

## 8. Recommended Exact Next Steps

If another agent resumes immediately, the safest order is:

1. Read this file completely.
2. Read the plan file completely.
3. Switch to the remediation worktree.
4. Recreate the symlink alias if it is missing.
5. Re-run the targeted Python regression suite to make sure the starting point still matches this handoff:

```bash
conda run -n executorch python -m pytest \
  examples/models/qwen3-tts/tests/test_convert_weights.py \
  examples/models/qwen3-tts/tests/test_startup_and_env.py \
  examples/models/qwen3-tts/tests/test_decoder_clone_parity.py \
  -q
```

6. If a C++ rebuild is needed, use the symlink-alias path and the exact commands from Section 5.3.
7. Continue with `runtime-hardening` first.

Suggested first code slice for `runtime-hardening`:

- create/extend tests first for malformed codec and manifest handling
- replace `export_manifest.json` parsing with `nlohmann/json`
- replace metadata sidecar parsing with `nlohmann/json`
- then harden codec input parsing
- then replace the fixed temp file
- then move to lazy bucket loading

## 9. Suggested Commands For The Next Agent

### Resume Worktree

```bash
cd "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation"
git status --short --branch
```

### Recreate Build Alias If Missing

```bash
cd "/Users/younghan/project/executorch"
ln -s "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation" ".worktrees/executorch"
```

### Sync Submodules In The Remediation Worktree

```bash
git -C "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation" submodule update --init --recursive
```

### Re-run Verified Python Tests

```bash
cd "/Users/younghan/project/executorch/.worktrees/qwen3-tts-red-team-remediation"
conda run -n executorch python -m pytest \
  examples/models/qwen3-tts/tests/test_convert_weights.py \
  examples/models/qwen3-tts/tests/test_startup_and_env.py \
  examples/models/qwen3-tts/tests/test_decoder_clone_parity.py \
  -q
```

### Rebuild Runner Using Working Path

```bash
cmake -S "/Users/younghan/project/executorch/.worktrees/executorch" --preset llm-release
cmake --build "/Users/younghan/project/executorch/.worktrees/executorch/cmake-out" --target install --parallel 8
cmake --preset qwen3-tts-cpu -S "/Users/younghan/project/executorch/.worktrees/executorch/examples/models/qwen3-tts"
cmake --build "/Users/younghan/project/executorch/.worktrees/executorch/cmake-out/examples/models/qwen3-tts" --target qwen3_tts_runner --parallel 8
```

## 10. Final Honesty Notes

- The code state is real and verified for the completed workstreams listed above.
- The fresh runner build verification is real and was done in this session using the symlink alias workaround.
- The next workstreams are **not** started yet, except for the todo tracker moving `runtime-hardening` to in progress.
- The current environment is still intentionally hostile to real `qwen_tts` execution until `transformers` and `sox` are corrected.
- The last code-review subagent for task 2 aborted because the session moved on, not because a code issue was found. The latest concrete state was validated by tests, lints, and runner build instead.
