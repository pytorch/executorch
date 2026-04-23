#!/usr/bin/env bash
# Voxtral TTS — CUDA end-to-end script.
# Exports the 4w-quantized full-CUDA pipeline (LM + codec both on GPU) and
# runs the runner. Total wall clock for "Hello, how are you today?" on A100:
# ~3.7 s (LM 2.1 s + codec 0.04 s + load/build).
#
# Usage:
#   conda activate et-cuda
#   unset CPATH                  # critical — see PROGRESS.md
#   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#   bash examples/models/voxtral_tts/run_cuda_e2e.sh \
#     <voxtral-model-dir> [<output-dir>]
#
# Env overrides:
#   SKIP_EXPORT=1 — skip export step (use existing artifacts in OUT_DIR)
#   SKIP_BUILD=1  — skip cmake build step
#   PROMPT="..."  — override the synthesis text
#   VOICE=<name>  — voice embedding name without .pt (default: neutral_female)
#   SEED=<int>    — RNG seed (default: 42)

set -euo pipefail

VOXTRAL_DIR="${1:?usage: $0 <voxtral-model-dir> [<output-dir>]}"
OUT_DIR="${2:-$PWD/voxtral_tts_exports_cuda_4w}"
PROMPT="${PROMPT:-Hello, how are you today?}"
VOICE="${VOICE:-neutral_female}"
SEED="${SEED:-42}"

if [[ -n "${CPATH:-}" ]]; then
  echo "ERROR: CPATH is set ('$CPATH'). Run 'unset CPATH' first." >&2
  echo "       It pollutes nvcc's include search and breaks the CUDA backend." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null; then
  echo "ERROR: nvcc not on PATH. Source ~/.bashrc and retry." >&2
  exit 1
fi
if [[ ! -d "$VOXTRAL_DIR" ]]; then
  echo "ERROR: model dir '$VOXTRAL_DIR' does not exist." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="$REPO_ROOT/cmake-out/examples/models/voxtral_tts/voxtral_tts_runner"

echo "=== 1/4. env check ==="
which nvcc
nvcc --version | tail -3 || true
echo "  CUDA_HOME=${CUDA_HOME:-unset}"
echo "  CUDAToolkit_ROOT=${CUDAToolkit_ROOT:-unset}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv

echo
echo "=== 2/4. export 4w-quant CUDA model + CUDA codec ==="
if [[ "${SKIP_EXPORT:-0}" == "1" && -f "$OUT_DIR/model.pte" ]]; then
  echo "  SKIP_EXPORT=1 and $OUT_DIR/model.pte exists — skipping export"
else
  mkdir -p "$OUT_DIR"
  python "$REPO_ROOT/examples/models/voxtral_tts/export_voxtral_tts.py" \
    --model-path "$VOXTRAL_DIR" \
    --backend cuda --qlinear 4w \
    --output-dir "$OUT_DIR"
fi
echo "  output:"
ls -la "$OUT_DIR"

echo
echo "=== 3/4. build voxtral_tts_runner with EXECUTORCH_BUILD_CUDA=ON ==="
if [[ "${SKIP_BUILD:-0}" == "1" && -x "$RUNNER" ]]; then
  echo "  SKIP_BUILD=1 and $RUNNER exists — skipping build"
else
  ( cd "$REPO_ROOT" && cmake --workflow --preset llm-release-cuda )
  ( cd "$REPO_ROOT/examples/models/voxtral_tts" && cmake --workflow --preset voxtral-tts-cuda )
fi

echo
echo "=== 4/4. synth: '$PROMPT' (voice=$VOICE seed=$SEED) ==="
WAV_OUT="${WAV_OUT:-$OUT_DIR/sample.wav}"
"$RUNNER" \
  --model "$OUT_DIR/model.pte" \
  --data_path "$OUT_DIR/aoti_cuda_blob.ptd" \
  --codec "$OUT_DIR/codec_decoder.pte" \
  --codec_data_path "$OUT_DIR/codec_aoti_cuda_blob.ptd" \
  --tokenizer "$VOXTRAL_DIR/tekken.json" \
  --voice "$VOXTRAL_DIR/voice_embedding/${VOICE}.pt" \
  --text "$PROMPT" \
  --output "$WAV_OUT" \
  --seed "$SEED" \
  --max_new_tokens 200

echo
echo "DONE. Wav: $WAV_OUT"
echo "      Listen: ffplay $WAV_OUT  (or aplay $WAV_OUT)"
