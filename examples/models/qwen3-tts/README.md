## Qwen3-TTS (XNNPACK)

This directory adds an ExecuTorch bring-up for
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` with an XNNPACK backend.

The pipeline has three stages, all exportable to ExecuTorch:

1. **Talker** (28-layer Qwen3 transformer): text → codec codes
2. **Code predictor** (5-layer sub-talker): predicts remaining 15 codebook
   groups per timestep
3. **Decoder** (vocoder): codec codes → audio waveform

### Performance

8da4w quantized, XNNPACK CPU (91 codes → 7.28s audio):

| Stage | Configuration | Time |
|---|---|---|
| Decoder | Bucket 150 (recommended) | **3.1s** |
| Decoder | Bucket 1200 (old default) | 32.4s |
| Decoder | Streaming 25-code chunks | 2.15s first audio, 6.68s total |
| Talker | 91 steps (max_seq=256) | 5.8s |
| Code predictor | 1365 steps (max_seq=32) | 9.8s |

Streaming decode emits first audio in **2.15s** by decoding 25-code chunks
incrementally instead of waiting for all codes.

## Files

- `convert_weights.py`: converts HF snapshot into decoder/talker artifacts.
- `convert_talker_weights.py`: converts talker weights to Meta/Llama format.
- `export_qwen3_tts.py`: exports decoder to ExecuTorch (bucketed).
- `export_talker.py`: exports talker/code predictor to ExecuTorch with KV cache.
- `generate_codes.py`: generates codec tokens from text (Python helper).
- `streaming_generate.py`: streaming decode with chunked vocoder inference.
- `main.cpp`, `qwen3_tts_runner.*`: C++ runner for decoder inference.
- `config/talker_config.json`: talker model config (Qwen3 Llama format).
- `config/code_predictor_config.json`: code predictor model config.

## Prerequisites

- ExecuTorch built from source.
- Conda env `executorch`.
- `qwen-tts` installed in that env.
- Access to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on Hugging Face.

## 1) Convert HF weights

```bash
python examples/models/qwen3-tts/convert_weights.py \
  Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  examples/models/qwen3-tts/qwen3_tts_artifacts \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --save-talker
```

Convert talker weights to Meta/Llama format:

```bash
python examples/models/qwen3-tts/convert_talker_weights.py \
  --talker-checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/qwen3_tts_talker.pth \
  --output-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted
```

## 2) Export decoder (8da4w bucketed)

```bash
python examples/models/qwen3-tts/export_qwen3_tts.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --backend xnnpack \
  --qlinear 8da4w \
  --bucket-sizes 75,150,300,600,1200 \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w_bucketed
```

This produces five `.pte` files (`model_75.pte` through `model_1200.pte`) and
an `export_manifest.json`. The bucket sizes correspond roughly to speech
durations of 6s, 12s, 25s, 50s, and 100s (12 codes/sec codec rate).

## 3) Export talker (8da4w)

Main talker (28-layer transformer with KV cache):

```bash
python examples/models/qwen3-tts/export_talker.py \
  --checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted/talker_main.pth \
  --params examples/models/qwen3-tts/config/talker_config.json \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_talker_8da4w \
  --backend xnnpack --qlinear 8da4w --max-seq-len 256
```

Code predictor (5-layer sub-talker):

```bash
python examples/models/qwen3-tts/export_talker.py \
  --checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted/talker_code_predictor.pth \
  --params examples/models/qwen3-tts/config/code_predictor_config.json \
  --output-dir examples/models/qwen3-tts/qwen3_tts_exports_talker_8da4w \
  --output-name code_predictor.pte \
  --backend xnnpack --qlinear 8da4w --max-seq-len 32
```

The talker uses the same Llama/Qwen3 infrastructure — architecturally identical
to Qwen3-0.6B with GQA, QK-norm, SiLU MLP, and RoPE.

## 4) Build runner

```bash
make qwen3-tts-cpu
```

## 5) Run decoder with bucketed models

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_runner \
  --model_dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w_bucketed \
  --text "Hello from ExecuTorch Qwen3 TTS." \
  --language English \
  --model_id_or_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --helper_script examples/models/qwen3-tts/generate_codes.py \
  --output_wav output.wav
```

The runner automatically selects the smallest bucket that fits the input.

## 6) Streaming decode from pre-generated codes

```bash
python examples/models/qwen3-tts/streaming_generate.py \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_exports_talker_8da4w \
  --decoder-dir examples/models/qwen3-tts/qwen3_tts_exports_8da4w_bucketed \
  --codes-path examples/models/qwen3-tts/metal_test_codes.bin \
  --output-wav output_streaming.wav \
  --chunk-size 25
```

Decodes audio incrementally in 25-code chunks, emitting first audio in ~2s
instead of waiting for all codes to be generated.

## Notes

- Dynamic-shape export is blocked by conv padding guard constraints in
  `torch.export`. Bucketed export is the workaround for the decoder.
- The talker uses static KV cache via the Llama infrastructure.
  `max_seq_len` strongly affects performance (256 recommended for typical use).
- All experiment commands and outcomes are tracked in `PROGRESS.md`.
