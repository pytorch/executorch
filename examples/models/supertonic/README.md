# Supertonic on ExecuTorch MLX

This example exports [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)
to one dynamic FP16 ExecuTorch program and performs one-shot or persistent
text-to-speech synthesis with the MLX delegate.

Run all commands from the ExecuTorch repository root. Keep downloaded assets
and generated programs outside the source tree:

```bash
export SUPERTONIC_ASSETS="${TMPDIR:-/tmp}/supertonic-3"
export SUPERTONIC_PTE="${SUPERTONIC_ASSETS}/supertonic_fp16_mlx.pte"
```

## 1. Download the model assets

Install this example's Python dependencies, then download the exact reviewed
Hugging Face revision:

```bash
python -m pip install -r examples/models/supertonic/requirements.txt

hf download Supertone/supertonic-3 \
  --revision 3cadd1ee6394adea1bd021217a0e650ede09a323 \
  --local-dir "${SUPERTONIC_ASSETS}"
```

The export reads `onnx/tts.json` and the four ONNX models. The native runner
also reads `onnx/unicode_indexer.json` and one JSON file under `voice_styles/`.

## 2. Export the ExecuTorch program

```bash
python -m examples.models.supertonic.export.export_supertonic \
  --asset-dir "${SUPERTONIC_ASSETS}" \
  --output "${SUPERTONIC_PTE}" \
  --max-text-length 512 \
  --max-latent-length 512 \
  --flow-steps 5
```

The generated PTE embeds the model weights; it does not require a `.ptd`
sidecar. Treat PTE files as trusted inputs: load only a PTE that you generated
or obtained from a trusted source.

## 3. Build the native runner

```bash
make supertonic-mlx
```

The runner is written to
`cmake-out/examples/models/supertonic/supertonic_runner`, with the required
`mlx.metallib` beside it.

## 4. Synthesize one WAV

```bash
./cmake-out/examples/models/supertonic/supertonic_runner \
  --pte="${SUPERTONIC_PTE}" \
  --asset_dir="${SUPERTONIC_ASSETS}" \
  --voice_style="${SUPERTONIC_ASSETS}/voice_styles/F1.json" \
  --text="Hello from Supertonic." \
  --language=en \
  --speed=1.05 \
  --seed=42 \
  --output="${SUPERTONIC_ASSETS}/hello.wav"
```

The runner writes a mono PCM16 WAV at the sample rate recorded in the model
metadata (44.1 kHz for the pinned assets).

## 5. Keep the runner warm

Use `--server_jsonl` to load the PTE once, perform one discarded warmup, and
process synthesis requests sequentially from stdin:

```bash
./cmake-out/examples/models/supertonic/supertonic_runner \
  --pte="${SUPERTONIC_PTE}" \
  --asset_dir="${SUPERTONIC_ASSETS}" \
  --voice_style="${SUPERTONIC_ASSETS}/voice_styles/F1.json" \
  --language=en \
  --speed=1.05 \
  --seed=42 \
  --server_jsonl
```

After loading and warmup, the runner emits this protocol-v1 `ready` schema;
`sample_rate` is always `44100` for the supported model:

```json
{"type":"ready","protocol_version":1,"sample_rate":44100,"load_seconds":0.03,"warmup_seconds":0.04}
```

Each request writes one complete WAV before returning its timing and RTF:

```json
{"type":"synthesize","id":1,"text":"Hello.","output":"/tmp/supertonic/1.wav"}
{"type":"result","id":1,"output":"/tmp/supertonic/1.wav","samples":82810,"audio_seconds":1.8778,"synthesis_seconds":0.0524,"rtf":0.0279}
```

Stdout is reserved for JSONL responses and flushed after every record;
diagnostics go to stderr. Request lines are limited to 64 KiB, and IDs must be
positive and strictly increasing. Output files are created atomically and must
not already exist. Malformed requests return an `error` record without stopping
the runner. Send `{"type":"shutdown"}` to receive `{"type":"stopped"}` and
exit cleanly; closing stdin also exits with status zero.

## Platform and model limits

- This workflow requires an Apple silicon Mac, macOS, Xcode command-line
  tools, CMake 3.26 or newer, and an ExecuTorch Python environment with the MLX
  backend and custom operations available. The native runner supports only
  arm64 Darwin and uses MLX GPU delegation with FP16 activations.
- Exported programs use dynamic sequence lengths, five flow-matching steps,
  batch size 1, and exactly one voice style.
- The commands above export maximum text and latent lengths of 512. A single
  sentence is never split, so a sentence that exceeds the exported text bound
  is rejected.
- Language tags and voice styles are limited to those provided by the pinned
  Supertonic 3 release.

## Provenance and licensing

The Supertonic architecture and portions of this integration are adapted from
[Supertone's Supertonic repository](https://github.com/supertone-inc/supertonic)
at revision
[`7e2804f96016a7028cb1ed627353c61c1e9dd281`](https://github.com/supertone-inc/supertonic/commit/7e2804f96016a7028cb1ed627353c61c1e9dd281),
which is licensed under the MIT License. The complete upstream copyright and
license notice is preserved in [`NOTICE`](NOTICE). ExecuTorch-specific code is
licensed under the BSD-style license in the repository root.

Model weights, voice styles, configuration, and other assets downloaded from
[Hugging Face revision
`3cadd1ee6394adea1bd021217a0e650ede09a323`](https://huggingface.co/Supertone/supertonic-3/tree/3cadd1ee6394adea1bd021217a0e650ede09a323)
are separately licensed under the BigScience Open RAIL-M License included with
that release. This repository does not redistribute those assets. Do not add
the downloaded weights or an exported PTE containing them to the ExecuTorch
repository; obtain the assets from Hugging Face and review their license and use
restrictions before use or distribution.
