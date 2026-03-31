# Qwen3-TTS Metal Backend Benchmark

## Results

### Metal/AOTI Export ✅ WORKING

Export command:
```bash
python3 export_unified.py --backend metal --dtype fp32 \
  --converted-dir qwen3_tts_artifacts \
  --talker-dir qwen3_tts_artifacts/talker_converted \
  --output-dir /tmp/qwen3_tts_metal_v2
```

| Metric | Value |
|--------|-------|
| Export time | ~8 min (AOTInductor compile) |
| Model size | 4,636 MB (fp32, no quantization) |
| Methods | 7 (encode_text, talker, code_predictor, codec_embed, cp_head, cp_generate, decode_audio) |
| Metal methods | 5 (everything except codec_embed + decode_audio) |
| decode_audio backend | XNNPACK (Metal lacks cumsum fallback) |

### Decode Performance (codes → audio)

| Backend | 26 codes | Realtime | Notes |
|---------|----------|----------|-------|
| Metal + XNNPACK decoder | **728 ms** | **2.86x RT** | Mixed: Metal talker, XNNPACK decoder |
| XNNPACK only | 1,056 ms | 2.42x RT | Previous best |
| Portable CPU (no backend) | 72,761 ms | 0.03x RT | When decoder has no XNNPACK |

### Audio Quality
- Metal output is correct: 1.59s speech from "Hello from ExecuTorch."
- Automatic silence trimming works

### Known Issues
1. `decode_audio` cannot use Metal (missing `cumsum` fallback kernel)
2. `fpa4w` quantization requires `TORCHAO_BUILD_EXPERIMENTAL_MPS=1`
3. libomp symlink needed: `sudo ln -sf /opt/homebrew/Cellar/libomp/*/lib/libomp.dylib /opt/llvm-openmp/lib/libomp.dylib`

### How to Run

```bash
# 1. Export (one time, ~8 min)
python3 examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir /tmp/qwen3_tts_metal \
  --backend metal --dtype fp32

# 2. Generate codes (Python talker)
python3 examples/models/qwen3-tts/generate_codes.py \
  --model-id-or-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "Your text here." \
  --output-codes /tmp/codes.bin

# 3. Decode (C++ Metal runner)
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path /tmp/qwen3_tts_metal/model.pte \
  --codes_path /tmp/codes.bin \
  --output_wav output.wav
```
