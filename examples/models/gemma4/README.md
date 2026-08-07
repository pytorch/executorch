# Gemma 4 on ExecuTorch

Multimodal inference for Gemma 4 on ExecuTorch.
Supports audio transcription, translation, image understanding, and text generation on mobile devices.

Variants: E2B (2B params) and E4B (4B params).

## Architecture

Single PTE with up to 4 methods:
- `speech_transform` — Waveform to log-mel spectrogram (no learned weights)
- `audio_encoder` — USM Conformer via HF's Gemma4AudioModel
- `vision_encoder` — ViT with 2D RoPE via HF's Gemma4VisionModel (8-bit, int8 position embeddings)
- `text_decoder` — Autoregressive decoder with YOCO, PLE, partial RoPE

Use `--no-audio` or `--no-vision` at export time to exclude unused encoders.

| | E2B | E4B |
|---|---|---|
| Hidden size | 1536 | 2560 |
| Layers | 35 | 42 |
| KV heads | 1 (MQA) | 2 |

## Export

```bash
# E2B default (4-bit text, 8-bit vision, all modalities):
buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
    --checkpoint_path /tmp/gemma4-e2b-it

# E2B 4-bit with tied embedding (smaller, for on-device deployment):
buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
    --checkpoint_path /tmp/gemma4-e2b-it --tied_embedding

# E4B (4-bit):
buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
    --checkpoint_path /tmp/gemma4-e4b-it --variant e4b

# Audio-only (no vision encoder, saves ~129 MB):
buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
    --checkpoint_path /tmp/gemma4-e2b-it --no-vision

# Vision-only (no audio encoder, saves ~100 MB):
buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
    --checkpoint_path /tmp/gemma4-e2b-it --no-audio
```

### Plain E2B WebGPU

The WebGPU path is independently exportable and text-only. It preserves an
8960-token KV capacity while bounding each input call to 512 tokens, returns a
delegated `Long[1, 1]` greedy token, and splits external constants into three
ordered PTD files below the browser binding/fetch limit. The default XNNPACK
export and runner are unchanged.

Acquire the checkpoint from
`google/gemma-4-E2B-it-qat-q4_0-unquantized` at immutable revision
`6befbaca7398925921802abd1f277b495b78b738`, then validate every staged byte:

```bash
hf download google/gemma-4-E2B-it-qat-q4_0-unquantized \
    model.safetensors config.json tokenizer.json tokenizer_config.json \
    generation_config.json processor_config.json chat_template.jinja \
    README.md .gitattributes \
    --revision 6befbaca7398925921802abd1f277b495b78b738 \
    --local-dir /tmp/gemma4-e2b-it
```

```bash
buck2 run fbcode//executorch/examples/models/gemma4:webgpu_artifact_manifest -- \
    validate-acquisition --checkpoint-root /tmp/gemma4-e2b-it
```

From clean fbsource and ExecuTorch OSS checkouts, seal the reviewed plain-Gemma
source union and generator-derived WGSL closure before exporting the model:

```bash
: "${FBSOURCE_ROOT:?set the clean fbsource checkout root}"
: "${OSS_ROOT:?set the clean ExecuTorch OSS checkout root}"
python -m executorch.examples.models.gemma4.webgpu_artifact_manifest \
  create-source-manifest --fbsource-root "$FBSOURCE_ROOT" \
  --oss-root "$OSS_ROOT" \
  --output /tmp/gemma4-source-manifest.json
python -m executorch.examples.models.gemma4.webgpu_artifact_manifest \
  create-wgsl-manifest \
  --backend-root "$FBSOURCE_ROOT/xplat/executorch/backends/webgpu" \
  --output /tmp/gemma4-wgsl-manifest.json
python -m executorch.examples.models.gemma4.webgpu_artifact_manifest \
  create-source-receipt \
  --fbsource-root "$FBSOURCE_ROOT" --oss-root "$OSS_ROOT" \
  --backend-root "$FBSOURCE_ROOT/xplat/executorch/backends/webgpu" \
  --output /tmp/gemma4-source-receipt.json
```

Export plain Gemma 4:

```bash
buck2 run fbcode//executorch/examples/models/gemma4:export_gemma4 -- \
    --checkpoint_path /tmp/gemma4-e2b-it \
    --output_path /tmp/gemma4-webgpu/model.pte \
    --backend webgpu --quantize 8da4w+emb4 \
    --max_seq_len 8960 --max_input_len 512 \
    --no-audio --no-vision \
    --source_receipt_path /tmp/gemma4-source-receipt.json \
    --artifact_manifest_output /tmp/gemma4-e2b-webgpu.json
```

The source-closure gate writes `gemma4-source-receipt.json` before this export.
The exporter reads the actual tensor-data insertion order, writes all three
content-named PTDs, and creates the manifest without renaming or globbing them.
Keep the manifest output outside the flat artifact staging directory.

```bash
buck2 run fbcode//executorch/examples/models/gemma4:webgpu_artifact_manifest -- \
    validate --root /tmp/gemma4-webgpu \
    --manifest /tmp/gemma4-e2b-webgpu.json
```

`manifests/gemma4_e2b_webgpu.json` pins the accepted ctx8960 behavior-oracle
quartet, but marks its old-worktree source closure as pending. It cannot satisfy
the production validator because it has no final-source receipt. Rebuild it
from the reviewed stack before claiming source-current performance or
reproduction. Dashboard and internal publication paths are evidence only and
are never source dependencies.

## Model Variants

Default export includes all modalities (audio + vision + text). Default context length: 1024 tokens (`--max_seq_len`).

### Pre-exported Models

**E2B:**

| File | Size | Config | Description |
|------|------|--------|-------------|
| `gemma4.pte` | 4.1 GB | 4-bit, audio-only | Default — fastest |
| `gemma4_vision.pte` | 4.3 GB | 4-bit, all modalities | Audio + vision + text |
| `gemma4_tied_emb4.pte` | 2.5 GB | 4-bit tied + emb4, audio-only | Smallest |

**E4B:**

| File | Size | Config | Description |
|------|------|--------|-------------|
| `gemma4.pte` | 6.1 GB | 4-bit, audio-only | Default — fastest |
| `gemma4_vision.pte` | 6.2 GB | 4-bit, all modalities | Audio + vision + text |
| `gemma4_tied_emb4.pte` | 4.0 GB | 4-bit tied + emb4, audio-only | Smallest |

### Export Flags

| Variant | Size | Flag |
|---------|------|------|
| E2B 4-bit (default) | 4.3 GB | (none) |
| E2B 4-bit audio-only | 4.1 GB | `--no-vision` |
| E2B 4-bit emb4 tied | 2.5 GB | `--quantize 8da4w+emb4 --tied_embedding --no-vision` |
| E4B 4-bit | 6.2 GB | `--variant e4b` |
| E4B 4-bit audio-only | 6.1 GB | `--variant e4b --no-vision` |
| E4B 4-bit emb4 tied | 4.0 GB | `--variant e4b --quantize 8da4w+emb4 --tied_embedding --no-vision` |

Vision encoder adds ~129 MB (8-bit linears + int8 position embedding table).

- **Untied models** (`gemma4.pte`, `gemma4_vision.pte`) work with both Python and C++ runners.
- **emb4 tied** uses packed INT4 embeddings and shared embed_tokens/lm_head weights. Requires C++ runner with TorchAO shared embedding kernels.

## Build (CMake, host)

```bash
cmake --preset gemma4-cpu -S examples/models/gemma4
cmake --build --preset gemma4-cpu -j$(nproc)
```

## Run

```bash
# Audio transcription (C++ runner):
./cmake-out/examples/models/gemma4/gemma4_e2e_runner \
    --model_path gemma4.pte \
    --tokenizer_path tokenizer.model \
    --audio_path test_audio.wav

# Image understanding (C++ runner):
./cmake-out/examples/models/gemma4/gemma4_e2e_runner \
    --model_path gemma4.pte \
    --tokenizer_path tokenizer.model \
    --image_path photo.jpg \
    --prompt "Describe this image:"

# Text-only:
./cmake-out/examples/models/gemma4/gemma4_e2e_runner \
    --model_path gemma4.pte \
    --tokenizer_path tokenizer.model \
    --prompt "What is 2+2?"

# Python runner (audio):
buck2 run fbcode//executorch/examples/models/gemma4:run_gemma4 -- \
    --model_path /tmp/gemma4.pte \
    --tokenizer_path /tmp/tokenizer.model \
    --audio_path /tmp/test_audio.wav

# Python runner (image):
buck2 run fbcode//executorch/examples/models/gemma4:run_gemma4 -- \
    --model_path /tmp/gemma4.pte \
    --tokenizer_path /tmp/tokenizer.model \
    --image_path /tmp/photo.jpg \
    --prompt "Describe this image:"
```

## Recommended Prompts

The runners default `--prompt` to a short generic string. For best output quality
on ASR / translation tasks, pass the canonical Google Gemma 4 prompt explicitly.

### Speech transcription (ASR)

```
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

Replace `{LANGUAGE}` with the source language (e.g., `English`, `Chinese`,
`Spanish`).

Example:

```bash
./gemma4_e2e_runner \
    --model_path gemma4.pte --tokenizer_path tokenizer.model \
    --audio_path test_audio.wav \
    --prompt "$(cat <<'EOF'
Transcribe the following speech segment in English into English text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
EOF
)"
```

### Speech translation

```
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

## Input Requirements

**Audio**: WAV, 16kHz, 16-bit PCM, mono, max 30 seconds.

**Image**: JPEG or PNG. Resized to fit `--max_vision_tokens` soft tokens (default 140). Aspect ratio preserved, dimensions rounded to multiples of 48 pixels. Lower tokens = faster but less detail (25 ~= 240x240, 70 ~= 384x384, 140 ~= 528x528, 280 ~= 768x768).

## Samsung S25 Performance

### Audio (23s)

| Model | Size | Load | Prefill | Gen | TTFT | RTF | Mem load | Mem peak |
|-------|------|------|---------|-----|------|-----|----------|----------|
| E2B gemma4.pte | 4.1 GB | 705ms | 166 tok/s | 6 tok/s | 4.50s | 0.71 | 1885 MB | 2251 MB |
| E2B gemma4_vision.pte | 4.3 GB | 648ms | 163 tok/s | 6 tok/s | 4.56s | 0.72 | 1890 MB | 2257 MB |
| E2B gemma4_tied_emb4.pte | 2.5 GB | 645ms | 164 tok/s | 6 tok/s | 4.52s | 0.71 | 1683 MB | 2241 MB |
| E4B gemma4.pte | 6.1 GB | 1.30s | 91 tok/s | 4 tok/s | 7.50s | 1.07 | 3231 MB | 3601 MB |
| E4B gemma4_vision.pte | 6.2 GB | 1.28s | 92 tok/s | 4 tok/s | 7.47s | 1.00 | 3231 MB | 3602 MB |
| E4B gemma4_tied_emb4.pte | 4.0 GB | 1.17s | 85 tok/s | 4 tok/s | 8.00s | 1.07 | 2899 MB | 3590 MB |

### Vision (dog.jpg, "Describe this image in two sentences.", 140 tokens ~528x528)

| Model | Size | Load | Encode | Prefill | Gen | TTFT | Total | Mem load | Mem peak |
|-------|------|------|--------|---------|-----|------|-------|----------|----------|
| E2B gemma4_vision.pte | 4.3 GB | 798ms | 2.73s | 134 tok/s | 6 tok/s | 3.83s | 10.14s | 1884 MB | 2600 MB |
| E4B gemma4_vision.pte | 6.2 GB | 1.36s | 2.44s | 85 tok/s | 4 tok/s | 4.17s | 14.62s | 3232 MB | 3950 MB |

### Text ("Write a short paragraph about the history of artificial intelligence")

| Model | Size | Load | Prefill | Gen | TTFT | Total | Mem load | Mem peak |
|-------|------|------|---------|-----|------|-------|----------|----------|
| E2B gemma4.pte | 4.1 GB | 625ms | 57 tok/s | 6 tok/s | 332ms | 26.94s | 1890 MB | 1950 MB |
| E4B gemma4.pte | 6.1 GB | 1.51s | 38 tok/s | 3 tok/s | 506ms | 44.66s | 3231 MB | 3287 MB |
