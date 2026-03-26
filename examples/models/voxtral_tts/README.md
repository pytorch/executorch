# Voxtral TTS

Self-contained ExecuTorch implementation of Mistral's Voxtral TTS,
a text-to-speech model that generates audio from text using a Mistral LLM
backbone with a flow-matching acoustic transformer and neural audio codec.
No HuggingFace Transformers dependency — weights are loaded directly from
the Mistral checkpoint.
See [model.md](model.md) for architecture and implementation details.

## Overview

The pipeline converts text to speech in two stages: **generation**
(autoregressive LLM + flow-matching acoustic transformer → audio codes)
and **decoding** (neural codec → waveform). Export produces two `.pte` files:

| File | Methods | Description |
|------|---------|-------------|
| `model.pte` | `token_embedding`, `text_decoder`, `lm_head`, `decode_audio_frame`, `audio_token_embedding` | LLM + acoustic transformer |
| `codec.pte` | `audio_decoder` | Neural audio codec decoder |

```
Text  ──→  LLM (Mistral)  ──→  hidden_states
                                     │
                                     └──→  AcousticTransformer (flow matching)  ──→  audio codes
                                                                                        │
                                     token_embedding(audio_tok=24)  ──→  embeds (fed back to LLM)

After generation:
  audio codes  ──→  CodecDecoder  ──→  waveform (.wav)
```

After the text prompt, the runner enters an audio generation loop where
every step runs `decode_audio_frame` on the LLM hidden state. The LM head
is not used — the acoustic transformer drives generation directly. The
flow-matching ODE runs 7 Euler steps with classifier-free guidance to
produce one frame of audio codes (1 semantic + N acoustic). The runner
feeds back the audio token (24) embedding for the next LLM step. Generation
stops when the acoustic transformer's semantic code signals END_AUDIO.
After generation completes, accumulated codes are decoded to a waveform
via the codec.

## Prerequisites

- ExecuTorch installed from source (see [building from source](../../../docs/source/using-executorch-building-from-source.md))
- [safetensors](https://pypi.org/project/safetensors/) (`pip install safetensors`)
- Model weights in Mistral format. The directory should contain `params.json`,
  `consolidated.safetensors`, and `tekken.json`.

## Export

Export converts the Mistral checkpoint into `.pte` files. The codec decoder
is exported separately with static shapes (chunked processing).

### Metal (recommended for Apple Silicon)

```bash
python export_voxtral_tts.py \
    --model-path ~/models/VoxtralTTS \
    --backend metal \
    --dtype bf16 \
    --output-dir ./voxtral_tts_exports
```

With quantization:

```bash
python export_voxtral_tts.py \
    --model-path ~/models/VoxtralTTS \
    --backend metal \
    --dtype bf16 \
    --output-dir ./voxtral_tts_exports \
    --qlinear fpa4w
```

Metal 4-bit quantization (`fpa4w`) requires torchao built with experimental MPS ops:

```bash
# From the ao repo (third-party/ao/)
USE_CPP=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 pip install . --no-build-isolation

# Or while installing ExecuTorch from source
EXECUTORCH_BUILD_KERNELS_TORCHAO=1 TORCHAO_BUILD_EXPERIMENTAL_MPS=1 ./install_executorch.sh
```

<details>
<summary><strong>XNNPACK (CPU)</strong></summary>

```bash
python export_voxtral_tts.py \
    --model-path ~/models/VoxtralTTS \
    --backend xnnpack \
    --output-dir ./voxtral_tts_exports \
    --qlinear 8da4w \
    --qembedding 8w
```

</details>

> [!NOTE]
> Use `--skip-codec` for faster iteration when only testing the LLM pipeline.

### Export Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Directory with `params.json` + `consolidated.safetensors` |
| `--backend` | `metal` | `metal` or `xnnpack` |
| `--dtype` | `fp32` | Model dtype: `fp32` or `bf16` |
| `--output-dir` | `./voxtral_tts_exports` | Output directory |
| `--max-seq-len` | `4096` | KV cache length |
| `--qlinear` | (none) | Linear layer quantization (`8da4w` for XNNPACK, `fpa4w` for Metal) |
| `--qlinear-group-size` | `32` | Group size for linear quantization |
| `--qembedding` | (none) | Embedding layer quantization (`8w`) |
| `--skip-codec` | off | Skip codec decoder export (produces `model.pte` only) |
| `--codec-chunk-size` | `375` | Static chunk size for codec decoder in frames |

**Notes:**
- `fpa4w` quantization requires `--backend metal`.
- `--dtype bf16` is recommended for Metal when using quantization.

## Build

ExecuTorch must be installed from source first (see
[Prerequisites](#prerequisites)). The `make` targets below handle
building core libraries and the runner binary.

```bash
make voxtral_tts-cpu      # XNNPACK (CPU)
make voxtral_tts-metal    # Metal (Apple GPU)
```

All targets produce the runner binary at
`cmake-out/examples/models/voxtral_tts/voxtral_tts_runner`.

## Run

The runner requires:
- `model.pte` — exported model (see [Export](#export))
- `tekken.json` — tokenizer from the model weights directory
- `codec.pte` — exported codec decoder (optional; without it, only audio codes are produced)

### Basic usage

```bash
cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model_path voxtral_tts_exports/model.pte \
    --codec_path voxtral_tts_exports/codec.pte \
    --tokenizer_path ~/models/VoxtralTTS/tekken.json \
    --prompt "Hello, how are you?" \
    --output_path output.wav
```

### Runner Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `model.pte` | Path to exported model |
| `--codec_path` | (none) | Path to codec decoder `.pte` (optional) |
| `--tokenizer_path` | `tekken.json` | Path to Tekken tokenizer |
| `--prompt` | (required) | Input text to synthesize |
| `--voice_path` | (none) | Path to voice embedding `.bin` file (see below) |
| `--output_path` | `output.wav` | Output WAV file path |
| `--max_tokens` | `2048` | Maximum tokens to generate |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--seed` | `42` | Random seed for flow matching noise |

## Exported Methods

### model.pte

| Method | Input | Output | Dynamic shapes |
|--------|-------|--------|----------------|
| `token_embedding` | `token_ids (1, seq)` | `embeds (1, seq, D)` | seq: `[1, max_seq_len]` |
| `text_decoder` | `embeds (1, seq, D)` + `cache_pos (seq,)` | `hidden (1, seq, D)` | seq: `[3, max_seq_len]` (Metal) or `[1, max_seq_len]` (XNNPACK) |
| `lm_head` | `hidden (1, 1, D)` | `logits (1, 1, vocab)` | static |
| `decode_audio_frame` | `hidden (1, D)` + `noise (1, C)` | `codes (1, 1+C)` | static |
| `audio_token_embedding` | `codes (1, K, seq)` | `embeds (1, seq, D)` | seq: `[1, max_seq_len]` |

### codec.pte

| Method | Input | Output | Dynamic shapes |
|--------|-------|--------|----------------|
| `audio_decoder` | `codes (1, K, T)` | `waveform (1, 1, T*upsample)` | static (T = chunk_size) |

## Inference Flow

```
1. token_embedding(prompt_tokens + voice_context) → embeds
2. Inject voice embeddings at audio token positions
3. text_decoder(embeds, positions) → hidden          # prefill
4. Audio generation loop (every step):
   a. text_decoder(token_emb(24), pos) → hidden      # single-step decode
   b. decode_audio_frame(hidden, randn()) → codes     # flow matching ODE
   c. If semantic_code == END_AUDIO: break
   d. Accumulate codes
5. audio_decoder(codes[:chunk]) → waveform            # for each chunk
6. Concatenate chunks → final audio → write WAV
```

## License

The ExecuTorch example code in this directory is licensed under the
BSD-style license found in the repository root.

The [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
model weights and bundled voice embeddings are licensed by Mistral AI under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(Creative Commons Attribution-NonCommercial 4.0 International). This means:

- **Attribution required**: credit Mistral AI and the voice data sources
  (EARS, CML-TTS, IndicVoices-R, Arabic Natural Audio datasets).
- **NonCommercial only**: the model weights may not be used for commercial
  purposes.

The NC restriction originates from the voice reference datasets used to
train the model. See the
[model card](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) for
full details.

## Troubleshooting

- **OOM during export**: Reduce `--max-seq-len` or use `--skip-codec` to
  export only `model.pte`.
- **`fpa4w` error**: This quantization requires `--backend metal`.
- **Metal runner fails with `Library not loaded: @rpath/libc++.1.dylib`**:
  The AOTInductor-compiled `.so` inside the `.pte` references `libc++` via
  `@rpath`. Add `/usr/lib` to `DYLD_LIBRARY_PATH`:
  ```bash
  DYLD_LIBRARY_PATH=/usr/lib \
      cmake-out/examples/models/voxtral_tts/voxtral_tts_runner ...
  ```
- **Metal runner fails with `Library not loaded: libomp.dylib`**:
  The AOTInductor-compiled `.so` links against OpenMP. Install it via
  Homebrew and add it to `DYLD_LIBRARY_PATH`:
  ```bash
  brew install libomp
  DYLD_LIBRARY_PATH=/usr/lib:$(brew --prefix libomp)/lib \
      cmake-out/examples/models/voxtral_tts/voxtral_tts_runner ...
  ```
- **`seq_len` constraint violation on Metal**: The MPS SDPA kernel requires
  `seq_len >= 3`. For single-token decode, the runner pads the input
  to length 3.
- **No audio output (only codes)**: Provide `--codec_path` to enable
  waveform generation. Without it, the runner generates audio codes but
  cannot produce a WAV file.
