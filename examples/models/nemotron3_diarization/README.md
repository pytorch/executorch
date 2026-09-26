# Nemotron 3 Diarization

```bash
# From the repository root:
python -m pip install git+https://github.com/huggingface/transformers "librosa>=0.10"

# Choose mlx (Apple Silicon), xnnpack (CPU), or cuda (NVIDIA GPU).
BACKEND=mlx
# Export: BF16 by default. Add --dtype fp32 to use FP32.
python -m executorch.examples.models.nemotron3_diarization.export_nemotron \
  --hf-model nvidia/Nemotron-3-Diarization \
  --backend "$BACKEND" \
  --output-dir "nemotron_exports/$BACKEND"

# Build the matching runner.
make "nemotron3-diarization-${BACKEND/xnnpack/cpu}"

# Convert to mono 16 kHz PCM16 and run
ffmpeg -i audio.wav -ac 1 -ar 16000 -c:a pcm_s16le audio_16khz_mono.wav
DATA_PATH=""
if [ "$BACKEND" = cuda ]; then
  DATA_PATH="nemotron_exports/cuda/aoti_cuda_blob.ptd"
fi
cmake-out/examples/models/nemotron3_diarization/nemotron3_diarization_runner \
  --model_path="nemotron_exports/$BACKEND/nemotron3_diarization.pte" \
  --data_path="$DATA_PATH" \
  --audio_path=audio_16khz_mono.wav \
  --preset=offline \
  --output=segments.json
```

CUDA export requires a CUDA-enabled PyTorch installation and a CUDA toolkit.
The preprocessor runs on CPU with XNNPACK; `pre_encode` and `encode` run on CUDA,
with float32 inputs and outputs and BF16 (or FP32) model computation. Export
produces `nemotron3_diarization.pte` and `aoti_cuda_blob.ptd`; pass the latter
through `--data_path` as shown above.
