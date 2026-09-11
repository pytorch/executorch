# Whisper with the OpenVINO Backend

This example exports and runs [OpenAI Whisper](https://huggingface.co/openai/whisper-small) speech-to-text models on Intel hardware using ExecuTorch OpenVINO backend.

## Overview

The model is split into three separate programs so that the cross-attention K/V projections — which depend only on the encoder output — run once per utterance instead of once per generated token:

- **encoder.pte** — mel features → encoder hidden states
- **cross_kv.pte** — encoder hidden states → per-layer cross-attention K/V
- **decoder.pte** — token-by-token generation with a self-attention KV cache, consuming the pre-computed cross K/V as inputs

## Environment Setup

Follow the **Prerequisites** and **Setup** instructions in [backends/openvino/README.md](../../../backends/openvino/README.md) to set up the OpenVINO backend.

### Install dependencies

```bash
pip install -r requirements.txt
```

## Export the Model

```bash
python export_whisper.py \
    --model_id openai/whisper-small \
    --output_dir ./whisper_ov \
    --device GPU \
    --max_cache_length 448
```

This writes four files to `./whisper_ov/`:

- `encoder.pte`
- `cross_kv.pte`
- `decoder.pte`
- `metadata.json`

## Run Inference

```bash
python run_whisper.py \
    --model_dir ./whisper_ov \
    --use_sample_audio
```

Or with a custom 16kHz audio file:

```bash
python run_whisper.py \
    --model_dir ./whisper_ov \
    --audio /path/to/audio.wav
```

Run either script with `--help` for the full set of options.
