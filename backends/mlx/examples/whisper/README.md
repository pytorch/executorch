# Whisper MLX Export

Export and run [OpenAI Whisper](https://huggingface.co/openai/whisper-tiny) speech-to-text models on the MLX backend.

## Prerequisites

```bash
pip install transformers torchao soundfile datasets
```

## Export

The export script splits the model into three programs:

- **encoder.pte** — audio features → encoder hidden states
- **cross_kv.pte** — encoder hidden states → per-layer cross-attention K/V
- **decoder.pte** — token-by-token generation with self-attention KV cache

Export with int4 weight quantization:

```bash
python -m executorch.backends.mlx.examples.whisper.export_whisper \
    --model-id openai/whisper-tiny \
    --output-dir /tmp/whisper_mlx \
    --dtype bf16 \
    --qlinear 4w
```

### Export Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `openai/whisper-tiny` | HuggingFace model ID |
| `--output-dir` | `whisper_mlx` | Output directory for `.pte` files |
| `--max-decoder-seq-len` | `256` | Maximum decoder sequence length |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--qlinear` | None | Quantization for linear layers (`4w`, `8w`, `nvfp4`) |
| `--qembedding` | None | Quantization for embedding layers (`4w`, `8w`, `nvfp4`) |
| `--qlinear-group-size` | auto | Group size for linear quantization |
| `--qembedding-group-size` | auto | Group size for embedding quantization |


## Run

```bash
python -m executorch.backends.mlx.examples.whisper.run_whisper \
    --model-dir /tmp/whisper_mlx \
    --use-sample-audio
```

Or with a custom audio file:

```bash
python -m executorch.backends.mlx.examples.whisper.run_whisper \
    --model-dir /tmp/whisper_mlx \
    --audio-file /path/to/audio.wav
```

### Run Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | `/tmp/whisper_mlx` | Directory containing exported `.pte` files |
| `--model-id` | `openai/whisper-tiny` | HuggingFace model ID (used to load processor) |
| `--audio-file` | None | Path to audio file (WAV, MP3, etc.) |
| `--use-sample-audio` | off | Use sample audio from HuggingFace datasets |
| `--max-new-tokens` | `256` | Maximum tokens to generate |
| `--language` | `en` | Language code |
| `--task` | `transcribe` | `transcribe` or `translate` |
| `--dtype` | `bf16` | Input dtype (must match export dtype) |
