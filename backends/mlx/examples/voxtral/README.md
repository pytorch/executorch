# Voxtral MLX Export

Export [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
multimodal audio-language model to ExecuTorch with the MLX backend.

Uses [optimum-executorch](https://github.com/huggingface/optimum-executorch) for
the export pipeline.

## Prerequisites

```bash
pip install transformers torch optimum-executorch mistral-common librosa
```

## Export

Export with int4 weight quantization (recommended):

```bash
python -m executorch.backends.mlx.examples.voxtral.export_voxtral_hf \
    --output-dir voxtral_mlx \
    --dtype bf16 \
    --qlinear 4w
```

This produces:
- `model.pte` — the main model (audio_encoder, token_embedding, text_decoder)
- `preprocessor.pte` — mel spectrogram preprocessor for raw audio

### Export Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | `mistralai/Voxtral-Mini-3B-2507` | HuggingFace model ID |
| `--output-dir` | `voxtral_mlx` | Output directory |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--max-seq-len` | `1024` | Maximum sequence length for KV cache |
| `--max-audio-len` | `300` | Maximum audio length in seconds |
| `--qlinear` | `4w` | Linear layer quantization (`4w`, `8w`, `nvfp4`, or None) |
| `--qlinear-group-size` | auto | Group size for linear quantization |

### Quantization

The `4w` config uses int4 weight-only quantization with the HQQ algorithm for
optimal scale selection. This typically reduces model size by ~4x with minimal
quality loss.

## Run

Requires the C++ voxtral runner. Build with:

```bash
make voxtral-mlx
```

Run inference:

```bash
./cmake-out/examples/models/voxtral/voxtral_runner \
    --model_path voxtral_mlx/model.pte \
    --processor_path voxtral_mlx/preprocessor.pte \
    --tokenizer_path /path/to/tekken.json \
    --audio_path /path/to/audio.wav \
    --prompt "What is happening in this audio?" \
    --temperature 0
```

The `tekken.json` tokenizer is included in the model weights directory
downloaded from HuggingFace.
