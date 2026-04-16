# Voxtral-4B-TTS-2603 on ExecuTorch

Text-to-speech with [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) running on ExecuTorch.

## Architecture

Three-component pipeline generating 24kHz audio from text:

1. **Mistral LLM** (~4B params) — autoregressive text-to-hidden-states
2. **Flow Matching Head** (3-layer transformer) — hidden states to 37 audio codebook tokens per frame via 7-step Euler ODE
3. **Codec Decoder** (Conv1d/ConvTranspose1d + 8 transformer layers) — codebook tokens to waveform

## Quick Start

### 1. Export

```bash
# Download model
huggingface-cli download mistralai/Voxtral-4B-TTS-2603 --local-dir ~/models/Voxtral-4B-TTS-2603

# Export with 4-bit quantization for XNNPACK (recommended)
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend xnnpack \
    --qlinear 4w \
    --output-dir ./voxtral_tts_exports

# Export fp32 for portable (CPU) backend
python export_voxtral_tts.py \
    --model-path ~/models/Voxtral-4B-TTS-2603 \
    --backend portable \
    --output-dir ./voxtral_tts_exports
```

### 2. Build

```bash
# Build ExecuTorch first (if not already built)
cmake --preset et-release -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_BUILD_XNNPACK=ON
cmake --build cmake-out -j$(nproc)

# Build the runner
make voxtral_tts-cpu
# or: make voxtral_tts-xnnpack
```

### 3. Run

```bash
# Offline (full generation then decode)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model voxtral_tts_exports/model.pte \
    --codec voxtral_tts_exports/codec_decoder.pte \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --text "Hello, this is a test of Voxtral TTS on ExecuTorch." \
    --output output.wav

# Streaming (incremental codec decoding)
./cmake-out/examples/models/voxtral_tts/voxtral_tts_runner \
    --model voxtral_tts_exports/model.pte \
    --codec voxtral_tts_exports/codec_decoder.pte \
    --tokenizer ~/models/Voxtral-4B-TTS-2603/tekken.json \
    --text "Hello, this is a test." \
    --output output.wav \
    --streaming
```

## Backend Support

| Backend | Status | Quantization |
|---------|--------|-------------|
| CPU (portable) | Supported | fp32 |
| XNNPACK | Supported | 4w, 8w, 8da4w, 8da8w |

## Exported Artifacts

Two `.pte` files (like voxtral_realtime):

- **model.pte** — Multi-method: `token_embedding`, `text_decoder`, `semantic_head`, `predict_velocity`
- **codec_decoder.pte** — Audio codec decoder (Conv1d/ConvTranspose1d + transformers)

## Audio Parameters

- Sample rate: 24,000 Hz
- Frame rate: 12.5 Hz (1 codebook frame = 80ms audio)
- Codebooks: 37 per frame (1 semantic VQ-8192 + 36 acoustic FSQ-21)
- Flow matching: 7-step Euler ODE with classifier-free guidance (alpha=1.2)
