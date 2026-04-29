# Gemma4 (text) — ExecuTorch CUDA bring-up

First-pass bespoke export pipeline for Google's Gemma4-31B text path.
No quantization, no vision/audio, no HuggingFace dependency.

## Files

| File                | Purpose                                                              |
|---------------------|----------------------------------------------------------------------|
| `model.py`          | Self-contained `Gemma4TextModel` (sliding + full attention layers).  |
| `convert_weights.py`| HF safetensors checkpoint -> Gemma4TextModel state-dict mapping.     |
| `export.py`         | CLI for `torch.export` + CUDA backend lowering (decode + prefill).   |
| `inference.py`      | 6-stage validation: eager -> export -> lower -> numerical compare.   |

## Run the validation pipeline

Pipeline sanity check (random weights, no checkpoint):

```bash
cd /home/gasoonjia/executorch
python examples/models/gemma4/inference.py --tiny-test
```

Full Gemma4-31B (requires ~60 GB checkpoint and a GPU with enough memory):

```bash
cd /home/gasoonjia/executorch
python examples/models/gemma4/inference.py \
    --model-dir /home/gasoonjia/models/gemma-4-31B \
    --output-dir /tmp/gemma4_export \
    --max-seq-len 4096
```

Each stage prints `OK` or `FAILED` with a traceback. On the first failure
the script exits non-zero so CI can pick it up.

## Architecture notes

- 60 layers, layer_types = `[s,s,s,s,s,F] * 10` (every 6th is full).
- Sliding layers: head_dim=256, 16 KV heads, full RoPE (theta=10k), window=1024.
- Full layers: head_dim=512, 4 KV heads, **V reuses K** (`attention_k_eq_v=True`),
  partial RoPE (25% of dims rotated, theta=1M).
- Standard RMSNorm (NOT the Gemma2/3 unit-offset variant).
- 4 layer norms per decoder + a buffer `layer_scalar` (init 1.0).
- Logit softcapping `tanh(x/30) * 30`. Tied embeddings + lm_head.
