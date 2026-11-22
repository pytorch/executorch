# Pocket TTS

Export Kyutai's 100M-parameter English [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
(`b6369a24`, the January 2026 model) to XNNPACK for 24 kHz speech synthesis.
This example reuses [Phonon's](../phonon) streaming transformer, Mimi decoder,
KV-cache attention, and transposed-convolution rewrites. Pocket keeps its own
1024-dimensional transformer, 512-dimensional flow head, variance-based time
normalization, text embeddings, and generation defaults.
Its Mimi attention window includes the current position and 249 past positions.

From the ExecuTorch repository root, in a Python environment with ExecuTorch
installed from this checkout:

```bash
./install_executorch.sh
python -m pip install safetensors huggingface_hub sentencepiece

python -m executorch.examples.models.pocket_tts.export_pocket_tts \
  --backend xnnpack --output-dir pocket_tts_exports

python -m executorch.examples.models.pocket_tts.run_pocket_tts \
  --model pocket_tts_exports/pocket_tts.pte \
  --text "Hello from Pocket TTS on ExecuTorch." --output hello.wav
```

Export downloads the checkpoint, tokenizer, and Alba catalog voice from
[`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/tree/d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3)
at the pinned revision `d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3`.
For an offline export, pass `--checkpoint-dir` containing
`tts_b6369a24.safetensors`, `tokenizer.model`, and `embeddings/alba.safetensors`.
Other Pocket model revisions are not supported by this example.

The output directory contains `pocket_tts.pte`, `pocket_tts.json` (method I/O,
state shapes, constants, and delegation counts), `tokenizer.model`, and
`alba.safetensors`. By default the program is FP32 and contains three compute methods:

| Method | Inputs | Outputs |
| --- | --- | --- |
| `text_embed` | token IDs `[1, S]` | text embeddings `[1, S, 1024]` |
| `flow_lm` | embeddings `[1, S, 1024]`, start position `[1]`, noise `[1, 32]` | latent `[1, 32]`, EOS logit `[1, 1]`, next embeddings `[1, 1, 1024]` |
| `mimi_decode` | latent `[1, 1, 32]`, streaming state | PCM `[1, 1, 1920]`, updated state |

`S` ranges from 1 to 128. The FlowLM KV cache holds 1024 positions inside
the program. Prefill the voice's `audio_prompt` embeddings directly, then
the text embeddings, then generate from the `bos_embeds` constant method.
The runner splits prefill into batches, supplies Gaussian noise scaled by
`sqrt(temperature)`, and feeds back the generated embeddings. Starting at
position zero replaces the previous utterance's valid cache prefix. Mimi's
state is explicit and starts at zero for each utterance.

XNNPACK runs supported linear and convolution operations. FlowLM attention
uses `llama::sdpa_with_kv_cache`; the runtime must include the LLM custom
operators as well as XNNPACK and portable/optimized kernels, as in the Phonon
CPU build. Encoding new voices is outside this export; `--voice` accepts
another compatible catalog `.safetensors` file containing `audio_prompt`.

The Python runner handles one short utterance at a time and rejects text
whose estimated generation length exceeds the cache. `--temperature` defaults
to 0.7; use `--temperature 0` for deterministic comparison or `--seed` to
control sampling. It performs basic capitalization, punctuation, and short
prompt padding; it does not implement long-form sentence chunking or playback.

Quantization follows Phonon's FlowLM-only policy:

| Backend | `--quantize` | Linear scheme | Default weight group size |
| --- | --- | --- | --- |
| XNNPACK | `4bit` | dynamic INT8 activations, INT4 weights (`8da4w`) | 32 |
| XNNPACK | `8bit` | dynamic INT8 activations, INT8 weights (`8da8w`) | 0 (per-channel) |

```bash
python -m executorch.examples.models.pocket_tts.export_pocket_tts \
  --backend xnnpack --quantize 8bit --output-dir pocket_tts_xnnpack_int8
```

Only compatible linear layers in the FlowLM transformer and flow network are
quantized. Mimi, text embeddings, input projection, and the EOS head stay FP32,
as do method inputs, outputs, and streaming state. Override the weight group
size with `--quantize-group-size`; layers whose input dimensions are not
divisible by the group size are skipped. The contract records `quantize`,
`qlinear`, and the requested `quantize_group_size` (`null` uses the default).
Quantization can change sampled speech and EOS timing; compare audio on representative prompts
when choosing a scheme.

Tests use small random models and require no checkpoint downloads:

```bash
python -m pytest examples/models/pocket_tts/tests -q
```

The tests cover strict checkpoint loading, Pocket's time normalization,
interleaved-RoPE reference parity, streaming state, and loading and running
FP32, INT4 and INT8 XNNPACK programs. See [NOTICE](NOTICE) for upstream attribution and
the separate model license.

The full FP32 export was also checked against `pocket-tts==1.0.3` with Alba
and temperature zero. "The quick brown fox jumps over the lazy dog." produces
38 frames (3.04 seconds) in both runtimes. With identical inputs per step,
maximum latent error was `1.1e-5` and maximum PCM error was `3.1e-6`.
