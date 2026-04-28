#!/usr/bin/env bash
# Voxtral TTS — MLX (Apple Silicon) end-to-end script.
# Exports the bf16 + 4w-quantized MLX LM/flow pipeline and a portable codec
# decoder, then runs the runner.
#
# Usage:
#   conda activate et-mlx
#   bash examples/models/voxtral_tts/run_mlx_e2e.sh \
#     <voxtral-model-dir> [<output-dir>]
#
# Env overrides:
#   SKIP_EXPORT=1 — skip export step (use existing artifacts in OUT_DIR)
#   SKIP_BUILD=1  — skip cmake build step
#   PROMPT="..."  — override the synthesis text
#   VOICE=<name>  — voice embedding name without .pt (default: neutral_female)
#   SEED=<int>    — RNG seed (default: 42)
#   DTYPE=<dt>    — fp32 | bf16 (default: bf16 — fastest on MLX)
#   QLINEAR=<q>   — 4w | 8w | 8da4w | 8da8w | "" (default: 4w)
#   QEMBEDDING=<q>— 4w | 8w | "" (default: 8w)

set -euo pipefail

VOXTRAL_DIR="${1:?usage: $0 <voxtral-model-dir> [<output-dir>]}"
OUT_DIR="${2:-$PWD/voxtral_tts_exports_mlx_4w}"
PROMPT="${PROMPT:-Hello, how are you today?}"
VOICE="${VOICE:-neutral_female}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
QLINEAR="${QLINEAR-4w}"
QEMBEDDING="${QEMBEDDING-8w}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: MLX requires macOS (Apple Silicon). Host is '$(uname -s)'." >&2
  exit 1
fi
if [[ ! -d "$VOXTRAL_DIR" ]]; then
  echo "ERROR: model dir '$VOXTRAL_DIR' does not exist." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="$REPO_ROOT/cmake-out/examples/models/voxtral_tts/voxtral_tts_runner"

echo "=== 1/4. env check ==="
python -c "import sys, platform; print(f'python {sys.version.split()[0]} on {platform.platform()}')"
python -c "import executorch; print('executorch:', getattr(executorch, '__version__', '?'))" || true
python -c "import executorch.backends.mlx  # noqa" || {
  echo "ERROR: executorch.backends.mlx is not importable. Re-run ./install_executorch.sh on macOS." >&2
  exit 1
}

echo
echo "=== 2/4. export ${DTYPE} + qlinear=${QLINEAR:-none} + qembedding=${QEMBEDDING:-none} MLX model + portable codec ==="
if [[ "${SKIP_EXPORT:-0}" == "1" && -f "$OUT_DIR/model.pte" && -f "$OUT_DIR/codec_decoder.pte" ]]; then
  echo "  SKIP_EXPORT=1 and $OUT_DIR/{model,codec_decoder}.pte exist — skipping export"
else
  mkdir -p "$OUT_DIR"
  EXPORT_ARGS=(
    --model-path "$VOXTRAL_DIR"
    --backend mlx
    --dtype "$DTYPE"
    --output-dir "$OUT_DIR"
  )
  if [[ -n "$QLINEAR" ]]; then
    EXPORT_ARGS+=(--qlinear "$QLINEAR")
  fi
  if [[ -n "$QEMBEDDING" ]]; then
    EXPORT_ARGS+=(--qembedding "$QEMBEDDING")
  fi
  python "$REPO_ROOT/examples/models/voxtral_tts/export_voxtral_tts.py" "${EXPORT_ARGS[@]}"
fi
echo "  output:"
ls -la "$OUT_DIR"

echo
echo "=== 3/4. build voxtral_tts_runner with EXECUTORCH_BUILD_MLX=ON ==="
if [[ "${SKIP_BUILD:-0}" == "1" && -x "$RUNNER" ]]; then
  echo "  SKIP_BUILD=1 and $RUNNER exists — skipping build"
else
  ( cd "$REPO_ROOT" && cmake --workflow --preset mlx-release )
  ( cd "$REPO_ROOT/examples/models/voxtral_tts" && cmake --workflow --preset voxtral-tts-mlx )
fi

echo
echo "=== 4/4. synth: '$PROMPT' (voice=$VOICE seed=$SEED) ==="
WAV_OUT="${WAV_OUT:-$OUT_DIR/sample.wav}"

# MLX may emit .ptd alongside the .pte when _tensor_data is populated. Include
# the flags only if the files exist so the runner works for either layout.
RUN_ARGS=(
  --model "$OUT_DIR/model.pte"
  --codec "$OUT_DIR/codec_decoder.pte"
  --tokenizer "$VOXTRAL_DIR/tekken.json"
  --voice "$VOXTRAL_DIR/voice_embedding/${VOICE}.pt"
  --text "$PROMPT"
  --output "$WAV_OUT"
  --seed "$SEED"
  --max_new_tokens 200
)
# Look for any .ptd file that pairs with model.pte / codec_decoder.pte.
MODEL_PTD="$(ls "$OUT_DIR"/*.ptd 2>/dev/null | grep -v codec_ | head -1 || true)"
CODEC_PTD="$(ls "$OUT_DIR"/codec_*.ptd 2>/dev/null | head -1 || true)"
if [[ -n "$MODEL_PTD" ]]; then
  RUN_ARGS+=(--data_path "$MODEL_PTD")
fi
if [[ -n "$CODEC_PTD" ]]; then
  RUN_ARGS+=(--codec_data_path "$CODEC_PTD")
fi

"$RUNNER" "${RUN_ARGS[@]}"

echo
echo "DONE. Wav: $WAV_OUT"
echo "      Listen: afplay $WAV_OUT  (or ffplay $WAV_OUT)"
