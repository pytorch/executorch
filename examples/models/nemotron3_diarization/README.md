# Nemotron 3 Diarization on MLX

```bash
# From the repository root on Apple Silicon:
python -m pip install git+https://github.com/huggingface/transformers "librosa>=0.10"

# Export (approximate BF16; add --dtype fp32 for comparison with Transformers)
# Uses the Transformers frontend.
python -m executorch.examples.models.nemotron3_diarization.export_nemotron \
  --hf-model nvidia/Nemotron-3-Diarization \
  --output-dir nemotron_exports

# Build
make nemotron3-diarization-mlx

# Convert to mono 16 kHz PCM16 and run
ffmpeg -i audio.wav -ac 1 -ar 16000 -c:a pcm_s16le audio_16khz_mono.wav
cmake-out/examples/models/nemotron3_diarization/nemotron3_diarization_runner \
  --model_path=nemotron_exports/nemotron3_diarization.pte \
  --audio_path=audio_16khz_mono.wav \
  --preset=offline \
  --output=segments.json
```
