# Whisper MLX Examples

Export and run [OpenAI Whisper](https://huggingface.co/openai/whisper-tiny) speech-to-text models on the MLX backend.

## Scripts

| Script | Description |
|---|---|
| `export_whisper.py` | Export with custom KV cache wrapper (3 separate `.pte` files) |
| `run_whisper.py` | Run models exported with `export_whisper` |

## Quick start

```bash
# Export
python -m executorch.backends.mlx.examples.whisper.export_whisper \
    --model-id openai/whisper-tiny \
    --output-dir /tmp/whisper_mlx

# Run
python -m executorch.backends.mlx.examples.whisper.run_whisper \
    --model-dir /tmp/whisper_mlx \
    --use-sample-audio
```


## export_whisper.py

Custom export that splits the model into three programs:

- **encoder.pte** — audio features → encoder hidden states
- **cross_kv.pte** — encoder hidden states → per-layer cross-attention K/V
- **decoder.pte** — token-by-token generation with self-attention KV cache

```bash
python -m executorch.backends.mlx.examples.whisper.export_whisper \
    --model-id openai/whisper-tiny \
    --output-dir /tmp/whisper_mlx \
    --quantize-linear int4
```

| Option | Default | Description |
|---|---|---|
| `--model-id` | `openai/whisper-tiny` | HuggingFace model ID |
| `--output-dir` | `whisper_mlx` | Output directory for `.pte` files |
| `--max-decoder-seq-len` | `256` | Maximum decoder sequence length |
| `--dtype` | `bf16` | Model dtype (`fp32`, `fp16`, `bf16`) |
| `--quantize-linear` | `None` | Quantize linear layers (`int4`, `int8`) |
| `--quantize-embeddings` | `None` | Quantize embedding layers (`int4`, `int8`) |
| `--linear-group-size` | `None` | Group size for linear quantization (32, 64, 128; default: 32 for int4, 128 for int8) |
| `--embeddings-group-size` | `None` | Group size for embedding quantization (32, 64, 128; default: 32 for int4, 128 for int8) |

## run_whisper.py

Run models exported with `export_whisper.py`. Loads encoder, cross_kv, and
decoder programs from a directory.

```bash
python -m executorch.backends.mlx.examples.whisper.run_whisper \
    --model-dir /tmp/whisper_mlx \
    --use-sample-audio
```

| Option | Default | Description |
|---|---|---|
| `--model-dir` | `/tmp/whisper_mlx` | Directory containing exported `.pte` files |
| `--model-id` | `openai/whisper-tiny` | HuggingFace model ID (used to load processor) |
| `--audio-file` | `None` | Path to audio file (WAV, MP3, etc.) |
| `--use-sample-audio` | `False` | Use sample audio from HuggingFace datasets |
| `--max-new-tokens` | `256` | Maximum tokens to generate |
| `--language` | `en` | Language code |
| `--task` | `transcribe` | `transcribe` or `translate` |
| `--dtype` | `bf16` | Input dtype (must match export dtype) |

## Requirements

After installing ExecuTorch, install optimum-executorch:

```bash
pip install optimum-executorch
pip install transformers torchao soundfile datasets
```
