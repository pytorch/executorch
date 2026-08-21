#!/usr/bin/env bash
# Export and run Voxtral Realtime on ROCm.
#
# Usage:
#   run_rocm_e2e.sh <model-dir> <wav> [bf16|w4-bf16|both] [streaming|offline|both] [output-root]
#
# Environment overrides:
#   SKIP_BUILD=1  Use an existing ROCm runner.
#   SKIP_EXPORT=1 Use existing model.pte and aoti_cuda_blob.ptd files.
#   DEVICE_INDEX  Visible GPU index (default: 0).
#   SLIDING_WINDOW Decoder window (default: 2048).
#   ROCM_PACKED_MATVEC=1 Use the experimental fixed-shape decoder matvec.
#   OFFLINE_MAX_NEW_TOKENS Offline token limit (default: 500).
#   VOXTRAL_PYTHON Python executable (default: python).
#   ROCM_PATH     ROCm installation (default: /opt/rocm).

set -euo pipefail

usage() {
  echo "usage: $0 <model-dir> <wav> [bf16|w4-bf16|both]" \
    "[streaming|offline|both] [output-root]" >&2
}

if [[ $# -lt 2 || $# -gt 5 ]]; then
  usage
  exit 1
fi

MODEL_DIR="$1"
AUDIO_PATH="$2"
PRECISION_MODE="${3:-w4-bf16}"
EXECUTION_MODE="${4:-streaming}"
OUTPUT_ROOT="${5:-$PWD/voxtral_rt_rocm}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
SLIDING_WINDOW="${SLIDING_WINDOW:-2048}"
ROCM_PACKED_MATVEC="${ROCM_PACKED_MATVEC:-0}"
OFFLINE_MAX_NEW_TOKENS="${OFFLINE_MAX_NEW_TOKENS:-500}"
VOXTRAL_PYTHON="${VOXTRAL_PYTHON:-python}"
ROCM_ROOT="${ROCM_PATH:-/opt/rocm}"

export HIP_VISIBLE_DEVICES="$DEVICE_INDEX"
export CUDA_VISIBLE_DEVICES="$DEVICE_INDEX"

if [[ "$ROCM_PACKED_MATVEC" != "0" && "$ROCM_PACKED_MATVEC" != "1" ]]; then
  echo "ERROR: ROCM_PACKED_MATVEC must be 0 or 1" >&2
  exit 1
fi

case "$PRECISION_MODE" in
  bf16) PRECISIONS=(bf16) ;;
  w4-bf16) PRECISIONS=(w4-bf16) ;;
  both) PRECISIONS=(bf16 w4-bf16) ;;
  *)
    echo "ERROR: precision must be bf16, w4-bf16, or both; got '$PRECISION_MODE'" >&2
    exit 1
    ;;
esac

case "$EXECUTION_MODE" in
  streaming) EXECUTIONS=(streaming) ;;
  offline) EXECUTIONS=(offline) ;;
  both) EXECUTIONS=(streaming offline) ;;
  *)
    echo "ERROR: execution mode must be streaming, offline, or both; got '$EXECUTION_MODE'" >&2
    exit 1
    ;;
esac

for required in params.json tekken.json consolidated.safetensors; do
  if [[ ! -f "$MODEL_DIR/$required" ]]; then
    echo "ERROR: missing $MODEL_DIR/$required" >&2
    exit 1
  fi
done
if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "ERROR: audio file '$AUDIO_PATH' does not exist" >&2
  exit 1
fi

"$VOXTRAL_PYTHON" - "$AUDIO_PATH" <<'PY'
import sys
import wave

with wave.open(sys.argv[1], "rb") as wav:
    actual = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
expected = (1, 2, 16000, "NONE")
if actual != expected:
    raise SystemExit(
        "ERROR: audio must be mono 16-bit PCM at 16 kHz; "
        f"got channels={actual[0]}, sample_width={actual[1]}, "
        f"sample_rate={actual[2]}, compression={actual[3]}"
    )
PY

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_ROOT="$REPO_ROOT/cmake-out-rocm-llm"
RUNNER="$BUILD_ROOT/examples/models/voxtral_realtime/voxtral_realtime_runner"
PYTHON_PREFIX="$("$VOXTRAL_PYTHON" -c 'import sys; print(sys.prefix)')"

monotonic_ms() {
  "$VOXTRAL_PYTHON" -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

"$VOXTRAL_PYTHON" - <<'PY'
import torch

if torch.version.hip is None:
    raise SystemExit("ERROR: the active Python environment is not a ROCm build")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no AMD GPU is visible to PyTorch")
print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))
PY

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  make -C "$REPO_ROOT" voxtral_realtime-rocm
elif [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: SKIP_BUILD=1 but runner is missing at $RUNNER" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib:$BUILD_ROOT/lib:$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_ROOT/preprocessors"
for execution in "${EXECUTIONS[@]}"; do
  preprocessor="$OUTPUT_ROOT/preprocessors/$execution.pte"
  if [[ "${SKIP_EXPORT:-0}" != "1" || ! -f "$preprocessor" ]]; then
    preprocessor_args=(--feature_size 128 --output_file "$preprocessor")
    if [[ "$execution" == "streaming" ]]; then
      preprocessor_args+=(--streaming)
    else
      preprocessor_args+=(--max_audio_len 300)
    fi
    "$VOXTRAL_PYTHON" -m executorch.extension.audio.mel_spectrogram \
      "${preprocessor_args[@]}"
  fi
done

for precision in "${PRECISIONS[@]}"; do
  quantization_args=()
  if [[ "$precision" == "w4-bf16" ]]; then
    quantization_args=(
      --qlinear-encoder 4w
      --qlinear 4w
      --qembedding 8w
    )
  fi

  for execution in "${EXECUTIONS[@]}"; do
    output_dir="$OUTPUT_ROOT/$precision/$execution"
    mkdir -p "$output_dir"

    export_args=(
      --model-path "$MODEL_DIR"
      --backend rocm
      --dtype bf16
      --output-dir "$output_dir"
    )
    if [[ "$execution" == "streaming" ]]; then
      export_args+=(--streaming --sliding-window "$SLIDING_WINDOW")
    fi
    if [[ "$precision" == "w4-bf16" && "$ROCM_PACKED_MATVEC" == "1" ]]; then
      export_args+=(--rocm-packed-matvec)
    fi

    export_elapsed_ms=-1
    if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
      export_start_ms="$(monotonic_ms)"
      TORCHINDUCTOR_CACHE_DIR="$OUTPUT_ROOT/torchinductor-$precision-$execution" \
        "$VOXTRAL_PYTHON" "$REPO_ROOT/examples/models/voxtral_realtime/export_voxtral_rt.py" \
          "${export_args[@]}" \
          "${quantization_args[@]}"
      export_end_ms="$(monotonic_ms)"
      export_elapsed_ms=$((export_end_ms - export_start_ms))
    elif [[ ! -f "$output_dir/model.pte" || ! -f "$output_dir/aoti_cuda_blob.ptd" ]]; then
      echo "ERROR: SKIP_EXPORT=1 but artifacts are missing in $output_dir" >&2
      exit 1
    fi

    "$VOXTRAL_PYTHON" - \
      "$output_dir/model.pte" \
      "$output_dir/aoti_cuda_blob.ptd" \
      "$precision" \
      "$execution" \
      "$export_elapsed_ms" <<'PY'
import pathlib
import sys

pte_path = pathlib.Path(sys.argv[1])
ptd_path = pathlib.Path(sys.argv[2])
precision = sys.argv[3]
execution = sys.argv[4]
elapsed_ms = int(sys.argv[5])
elapsed = "skipped" if elapsed_ms < 0 else f"{elapsed_ms / 1000:.3f}"
print(
    "Voxtral export metrics: "
    f"precision={precision} mode={execution} elapsed_seconds={elapsed} "
    f"pte_bytes={pte_path.stat().st_size} ptd_bytes={ptd_path.stat().st_size}"
)
PY

    runner_args=(
      --model_path "$output_dir/model.pte"
      --data_path "$output_dir/aoti_cuda_blob.ptd"
      --tokenizer_path "$MODEL_DIR/tekken.json"
      --preprocessor_path "$OUTPUT_ROOT/preprocessors/$execution.pte"
      --audio_path "$AUDIO_PATH"
      --temperature 0
    )
    if [[ "$execution" == "streaming" ]]; then
      runner_args+=(--streaming)
    else
      runner_args+=(--offline_max_new_tokens "$OFFLINE_MAX_NEW_TOKENS")
    fi
    runner_log="$output_dir/runner.log"
    "$RUNNER" "${runner_args[@]}" 2>&1 | tee "$runner_log"

    "$VOXTRAL_PYTHON" - \
      "$runner_log" \
      "$AUDIO_PATH" \
      "$precision" \
      "$execution" <<'PY'
import json
import pathlib
import sys
import wave

runner_log = pathlib.Path(sys.argv[1])
audio_path = sys.argv[2]
precision = sys.argv[3]
execution = sys.argv[4]

prefix = "PyTorchObserver "
stats_json = None
for line in runner_log.read_text(errors="replace").splitlines():
    if prefix in line:
        stats_json = line.split(prefix, 1)[1].strip()
if stats_json is None:
    raise SystemExit(f"ERROR: PyTorchObserver stats missing from {runner_log}")

stats = json.loads(stats_json)
inference_seconds = (
    stats["inference_end_ms"] - stats["inference_start_ms"]
) / stats["SCALING_FACTOR_UNITS_PER_SECOND"]
with wave.open(audio_path, "rb") as wav:
    audio_seconds = wav.getnframes() / wav.getframerate()
if audio_seconds <= 0:
    raise SystemExit(f"ERROR: audio duration must be positive: {audio_path}")
rtf = inference_seconds / audio_seconds
print(
    "Voxtral RTF: "
    f"precision={precision} mode={execution} "
    f"inference_seconds={inference_seconds:.3f} "
    f"audio_seconds={audio_seconds:.3f} rtf={rtf:.3f}"
)
PY
  done
done
