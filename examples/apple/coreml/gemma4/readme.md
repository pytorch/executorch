# Gemma 4 text decoder on CoreML

This directory exports the Gemma 4 text decoder shipped with
`examples/models/gemma4` to a CoreML-delegated ExecuTorch program.

Gemma 4's hybrid sliding/full attention, partial RoPE, per-layer
head_dim, MQA, and YOCO KV sharing are all expressed in plain PyTorch
in the upstream `examples/models/gemma4/text_decoder/` package, and that
implementation lowers cleanly through `torch.export` and
`CoreMLPartitioner` — every call is a single `executorch_call_delegate`
in the resulting `.pte`.  This script assembles the small amount of
glue (CoreML compile specs, iOS18+ deployment target for stateful KV
caches, fp16 conversion) needed to run that lowering with sensible
defaults for on-device deployment.

The audio and vision encoders are intentionally **not** exported here;
the existing ATen pipeline in `examples/models/gemma4` is more
appropriate for those.

## Usage

### From a HuggingFace checkpoint

```
python export_gemma4_text_decoder_coreml.py \
    --checkpoint_path /path/to/gemma4-e2b-it \
    --output gemma4_text_decoder.pte
```

### Synthetic config (smoke test, no weights)

```
python export_gemma4_text_decoder_coreml.py \
    --random_weights \
    --max_seq_len 1024 \
    --output /tmp/gemma4_synthetic.pte
```

## Options

| Option | Default | Description |
|---|---|---|
| `--checkpoint_path` | (required if no `--random_weights`) | HuggingFace Gemma 4 checkpoint dir |
| `--config_json` | (off) | Use this `config.json` instead of the checkpoint's |
| `--random_weights` | (off) | Skip weight loading; smoke-test only |
| `--max_seq_len` | 2048 | Maximum context length |
| `--input_len` | 64 | Prefill seqlen used for example inputs |
| `--sliding_window` | (from config) | Override sliding-attention window |
| `--sliding_window_pattern` | (from config) | Override hybrid pattern (P=5 for Gemma 4 E2B) |
| `--dtype` | `fp16` | `fp16` or `fp32`.  ANE requires fp16. |
| `--minimum_deployment_target` | `iOS18` | iOS17 / iOS18 / iOS26.  Stateful KV caches need iOS18+. |

## Tests

`test.py` builds a 10-layer synthetic Gemma 4 model (4 sliding + 1 full
× 2) and runs the full export pipeline, asserting that the resulting
`.pte` exists and is non-empty:

```
$ python -m pytest examples/apple/coreml/gemma4/test.py -v
============================== 2 passed in 15.32s ==============================
```
