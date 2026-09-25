# Nemotron 3 Diarization

```bash
# From the repository root:
python -m pip install git+https://github.com/huggingface/transformers "librosa>=0.10"

# Choose mlx (Apple Silicon), xnnpack (CPU), or vulkan (GPU).
BACKEND=mlx
# Export: BF16 for MLX/XNNPACK, FP32 for Vulkan. Add --dtype fp32 to use FP32.
python -m executorch.examples.models.nemotron3_diarization.export_nemotron \
  --hf-model nvidia/Nemotron-3-Diarization \
  --backend "$BACKEND" \
  --output-dir "nemotron_exports/$BACKEND"

# Build the matching runner (Vulkan requires the Vulkan SDK and glslc).
make "nemotron3-diarization-${BACKEND/xnnpack/cpu}"

# Convert to mono 16 kHz PCM16 and run
ffmpeg -i audio.wav -ac 1 -ar 16000 -c:a pcm_s16le audio_16khz_mono.wav
cmake-out/examples/models/nemotron3_diarization/nemotron3_diarization_runner \
  --model_path="nemotron_exports/$BACKEND/nemotron3_diarization.pte" \
  --audio_path=audio_16khz_mono.wav \
  --preset=offline \
  --output=segments.json
```
