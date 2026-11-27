# Reverb-TTS on ExecuTorch

This example exports [Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS),
a 297M-parameter voice-cloning text-to-speech model for English, German,
French, Spanish, Italian, Polish and Portuguese, to two ExecuTorch programs. A
native runner then speaks text in a preset voice, or in a voice cloned from a
few seconds of speech, on the MLX (Apple silicon GPU) or XNNPACK (CPU) backend.

Reverb is a masked-diffusion model: it fills a grid of audio-codec tokens over
several passes, conditioned on the text and on a reference voice's codec
tokens. The export splits the work as follows:

| Program | Method | Inputs | Outputs |
|---|---|---|---|
| `reverb.pte` | `prefill` | prompt token ids `[L]`, reference codes `[8, R]` | key/value cache `[2, 24, 6, L+R, 128]` |
| | `step` | codes `[8, T]`, cache | confidence, prediction and held log-probability of codebooks 0-1, each `[2, T]` |
| | `polish` | codes `[8, T]`, reference frames `[1]`, cache | codebooks 2-7 `[6, T]` |
| `codec.pte` | `decode` | codes `[8, F]` | 24 kHz waveform `[F * 960]` |
| | `encode` | 24 kHz waveform `[N]`, N a multiple of 960 | codes `[8, N / 960]` |

All lengths are dynamic. The runner prefills the prompt and the reference once,
runs eight main passes that commit the most confident codes of codebooks 0-1
(sampling with Gumbel noise and remasking codes the model comes to doubt), runs
both polish passes for codebooks 2-7 in one call, and decodes the codes to
audio. The codec is the
[Higgs Audio v2 tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer)
at 25 frames per second.

Run all commands from the ExecuTorch repository root. Keep downloaded assets
and generated programs outside the source tree:

```bash
export REVERB_ASSETS="${TMPDIR:-/tmp}/reverb-tts"
export CODEC_ASSETS="${TMPDIR:-/tmp}/higgs-audio-v2-tokenizer"
export REVERB_PTE_DIR="${TMPDIR:-/tmp}/reverb-pte"
```

## 1. Download the model assets

Install this example's Python dependencies, then download the reviewed
Hugging Face revisions of the model and the codec:

```bash
python -m pip install -r examples/models/reverb/requirements.txt

hf download software-mansion/Reverb-TTS \
  --revision 72be799c8fb6d56a206d260b425c48bf3bd5943d \
  --include "model/safetensors/fp32/*" --include "model/tokenizer.json" \
  --include "voices/*" \
  --local-dir "${REVERB_ASSETS}"

hf download bosonai/higgs-audio-v2-tokenizer \
  --revision 815a03f1b57dcb5ae918db415f22cd0c8a6f0f54 \
  --local-dir "${CODEC_ASSETS}"
```

## 2. Export the programs

```bash
python -m examples.models.reverb.export.export_reverb \
  --asset-dir "${REVERB_ASSETS}" \
  --codec-dir "${CODEC_ASSETS}" \
  --backend mlx \
  --output-dir "${REVERB_PTE_DIR}"
```

This writes `reverb.pte` and `codec.pte`. Each program embeds its weights.
Choose the backend, and optionally quantize the backbone's decoder layers:

| `--backend` | Default backbone dtype | `--qlinear` |
|---|---|---|
| `mlx` | fp16 | `4w`, `8w` |
| `xnnpack` | fp32 | `8da4w`, `8da8w` |
| `portable` | fp32 | none |

The codec stays fp32 unless you pass `--codec-dtype fp16`. The default bounds
take prompts of up to 512 tokens, references of up to 512 frames (20.5 s) and
takes of up to 1024 frames (41 s); change them with `--max-prompt-tokens`,
`--max-reference-frames` and `--max-target-frames`. Treat PTE files as trusted
inputs: load only a PTE that you generated or obtained from a trusted source.

## 3. Build the native runner

```bash
make reverb-mlx      # macOS: MLX and XNNPACK
make reverb-cpu      # XNNPACK
```

The runner is written to `cmake-out/examples/models/reverb/reverb_runner`; the
MLX build places `mlx.metallib` beside it.

## 4. Speak in a preset voice

```bash
./cmake-out/examples/models/reverb/reverb_runner \
  --backbone="${REVERB_PTE_DIR}/reverb.pte" \
  --codec="${REVERB_PTE_DIR}/codec.pte" \
  --tokenizer="${REVERB_ASSETS}/model/tokenizer.json" \
  --voice="${REVERB_ASSETS}/voices/en_female_narrator.json" \
  --language=en \
  --text="Hello from ExecuTorch. This is Reverb, speaking on device." \
  --seed=42 \
  --output="${REVERB_PTE_DIR}/hello.wav"
```

The runner writes a mono 16-bit WAV at 24 kHz and reports the time spent in
each program. Add `--warmup` to time a second take after a discarded first one.
The release's `voices/` folder holds 16 preset voices; a voice can speak any of
the seven languages.

## 5. Clone a voice

Give a WAV recording of one speaker (3 to 15 seconds of clean speech) and its
exact transcript:

```bash
./cmake-out/examples/models/reverb/reverb_runner \
  --backbone="${REVERB_PTE_DIR}/reverb.pte" \
  --codec="${REVERB_PTE_DIR}/codec.pte" \
  --tokenizer="${REVERB_ASSETS}/model/tokenizer.json" \
  --reference_audio=reference.wav \
  --reference_text="The exact words spoken in the recording." \
  --save_voice="${REVERB_PTE_DIR}/my_voice.json" \
  --text="Cloning a voice takes only a few seconds of speech." \
  --output="${REVERB_PTE_DIR}/cloned.wav"
```

The runner resamples the recording to 24 kHz, encodes it with `codec.pte` and,
with `--save_voice`, writes the voice in the preset format so later runs can
pass it to `--voice` without encoding again.

## Text preparation

The model reads text with numbers, dates, symbols and abbreviations written out
as words, as the training transcripts were. The runner speaks `--text` and the
reference transcript as given, so prepare them first, for example with the
[reverb-tts](https://pypi.org/project/reverb-tts/) package:

```python
from reverb_tts import prepare_text

print(prepare_text("It costs $5 on 3 May.", "en"))
```

## Platform and model limits

- Batch size is 1. A text estimated over 30 seconds is spoken in pieces of
  about 15 seconds, cut at sentence ends; a single sentence longer than the
  exported target bound is rejected.
- Sampling runs in the runner with a seeded Mersenne Twister, so a seed gives
  the same take on every platform, but not the take of the PyTorch package.
- The MLX runner requires an Apple silicon Mac.

## Tests

```bash
python -m pytest examples/models/reverb/tests -v
ctest --test-dir cmake-out/examples/models/reverb --output-on-failure
```

The Python tests build tiny random models and need no downloads. Set
`REVERB_ASSET_DIR` to a download of the release to also load its checkpoint.

## Provenance and licensing

The sampler, the target-length estimate and the reference-audio helpers are
adapted from [Reverb-TTS](https://github.com/software-mansion-labs/Reverb-TTS)
at revision
[`2c72a7f`](https://github.com/software-mansion-labs/Reverb-TTS/commit/2c72a7f9e9b7e8c9540d8823c7a8ef8833fd8346),
which is licensed under the Apache License 2.0; see [`NOTICE`](NOTICE).
ExecuTorch-specific code is licensed under the BSD-style license in the
repository root.

The Reverb-TTS weights, tokenizer and preset voices are licensed under
CC BY-NC 4.0, and the Higgs Audio v2 tokenizer weights under the Boson Higgs
Audio 2 Community License. This repository does not redistribute them. Do not
add the downloaded weights, or an exported PTE containing them, to the
ExecuTorch repository; review their licenses and use restrictions before use or
distribution.
