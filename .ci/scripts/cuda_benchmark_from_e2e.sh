#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run the CUDA perf benchmark for one (model, quant) pair, reusing the
# .pte/.ptd artifacts and the already-built runner left behind by
# test_model_e2e.sh in the same CI job.
#
# Only the (model, quant) pairs in the perf allowlist below are benchmarked.
# Anything else exits 0 after printing BENCHMARK_SKIP, so the caller can tell
# "not covered by perf" apart from a real failure. The optional
# models/quantizations filters (from workflow_dispatch) narrow the benchmark
# stage further without touching the accuracy stage.
#
# Usage:
#   cuda_benchmark_from_e2e.sh <hf_model> <quant> <model_dir> <results_dir>
#       [num_runs] [git_sha] [run_id] [run_url] [models_filter] [quants_filter]
#
# Outputs (in <results_dir>):
#   benchmark_results.json, benchmark_results_v3.json, metadata.json

set -euo pipefail

HF_MODEL="${1:?hf_model required, e.g. openai/whisper-small}"
QUANT_NAME="${2:?quant required, e.g. non-quantized}"
MODEL_DIR="${3:?model_dir required}"
RESULTS_DIR="${4:?results_dir required}"
NUM_RUNS="${5:-50}"
GIT_SHA="${6:-unknown}"
WORKFLOW_RUN_ID="${7:-0}"
WORKFLOW_RUN_URL="${8:-}"
MODELS_FILTER="${9:-}"
QUANTS_FILTER="${10:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# (model, quant) pairs covered by perf tracking. Keep in sync with the
# test-model-cuda-e2e matrix in .github/workflows/cuda.yml.
is_bench_pair() {
  case "${HF_MODEL} ${QUANT_NAME}" in
  "mistralai/Voxtral-Mini-3B-2507 "* | \
    "openai/whisper-small non-quantized" | \
    "openai/whisper-medium non-quantized" | \
    "openai/whisper-large-v3-turbo quantized-"* | \
    "google/gemma-3-4b-it non-quantized" | \
    "google/gemma-3-4b-it quantized-int4-tile-packed" | \
    "nvidia/parakeet-tdt non-quantized" | \
    "nvidia/parakeet-tdt quantized-int4-tile-packed" | \
    "SocialLocalMobile/Qwen3.5-35B-A3B-HQQ-INT4 quantized-int4-tile-packed")
    return 0
    ;;
  *)
    return 1
    ;;
  esac
}

if ! is_bench_pair; then
  echo "BENCHMARK_SKIP: ${HF_MODEL} ${QUANT_NAME} is not in the perf allowlist"
  exit 0
fi

# workflow_dispatch filters: comma-separated HF model IDs / quant names.
# Empty means "benchmark every allowlisted pair".
case ",${MODELS_FILTER}," in
*",${HF_MODEL},"*) ;;
*)
  if [ -n "${MODELS_FILTER}" ]; then
    echo "BENCHMARK_SKIP: ${HF_MODEL} not in dispatch models filter"
    exit 0
  fi
  ;;
esac
case ",${QUANTS_FILTER}," in
*",${QUANT_NAME},"*) ;;
*)
  if [ -n "${QUANTS_FILTER}" ]; then
    echo "BENCHMARK_SKIP: ${QUANT_NAME} not in dispatch quantizations filter"
    exit 0
  fi
  ;;
esac

cd "${EXECUTORCH_ROOT}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Runner command mapping, ported from the retired cuda-perf.yml so the
# benchmark prompts and flags (and therefore the numbers) stay identical.
case "${HF_MODEL}" in
mistralai/Voxtral-Mini-3B-2507)
  RUNNER="cmake-out/examples/models/voxtral/voxtral_runner"
  PREPROCESSOR="${MODEL_DIR}/voxtral_preprocessor.pte"
  TOKENIZER="${MODEL_DIR}/tekken.json"
  AUDIO="${MODEL_DIR}/poem.wav"
  RUNNER_CMD="$RUNNER --model_path ${MODEL_DIR}/model.pte --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd --tokenizer_path $TOKENIZER --audio_path $AUDIO --processor_path $PREPROCESSOR --temperature 0"
  MODEL_NAME="voxtral_${QUANT_NAME}"
  ;;
openai/whisper-*)
  RUNNER="cmake-out/examples/models/whisper/whisper_runner"
  PREPROCESSOR="${MODEL_DIR}/whisper_preprocessor.pte"
  AUDIO="${MODEL_DIR}/output.wav"
  RUNNER_CMD="$RUNNER --model_path ${MODEL_DIR}/model.pte --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd --tokenizer_path ${MODEL_DIR}/ --audio_path $AUDIO --processor_path $PREPROCESSOR --temperature 0"
  MODEL_NAME="${HF_MODEL#openai/}_${QUANT_NAME}"
  ;;
google/gemma-3-4b-it)
  RUNNER="cmake-out/examples/models/gemma3/gemma3_e2e_runner"
  IMAGE="docs/source/_static/img/et-logo.png"
  RUNNER_CMD="$RUNNER --model_path ${MODEL_DIR}/model.pte --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd --tokenizer_path ${MODEL_DIR}/ --image_path $IMAGE --temperature 0"
  MODEL_NAME="gemma3_${QUANT_NAME}"
  ;;
nvidia/parakeet-tdt)
  RUNNER="cmake-out/examples/models/parakeet/parakeet_runner"
  AUDIO="${MODEL_DIR}/test_audio.wav"
  TOKENIZER="${MODEL_DIR}/tokenizer.model"
  RUNNER_CMD="$RUNNER --model_path ${MODEL_DIR}/model.pte --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd --audio_path $AUDIO --tokenizer_path $TOKENIZER"
  MODEL_NAME="parakeet_${QUANT_NAME}"
  ;;
SocialLocalMobile/Qwen3.5-35B-A3B-HQQ-INT4)
  RUNNER="cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_runner"
  TOKENIZER="${MODEL_DIR}/tokenizer.json"
  # A checked-in long prompt (>1000 tokens). A static, meaningful prompt
  # avoids the degenerate / repetitive outputs that can result from
  # synthetic prompts built by repeating the same sentence.
  PROMPT_FILE=".ci/scripts/cuda_perf_prompts/qwen3_5_moe_long_prompt.txt"
  RUNNER_CMD="$RUNNER --model_path ${MODEL_DIR}/model.pte --data_path ${MODEL_DIR}/aoti_cuda_blob.ptd --tokenizer_path $TOKENIZER --prompt_file $PROMPT_FILE --max_new_tokens 512 --temperature 0"
  MODEL_NAME="qwen3_5_moe_${QUANT_NAME}"
  ;;
*)
  echo "BENCHMARK_SKIP: no runner mapping for '${HF_MODEL}'"
  exit 0
  ;;
esac

echo "::group::Running benchmark for ${HF_MODEL} (${QUANT_NAME}) with ${NUM_RUNS} runs"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "CUDA Driver Version: $CUDA_DRIVER_VERSION"

mkdir -p "${RESULTS_DIR}"

# Run benchmark using cuda_benchmark.py
python .ci/scripts/cuda_benchmark.py \
  --runner_command "$RUNNER_CMD" \
  --model_name "$MODEL_NAME" \
  --num_runs "${NUM_RUNS}" \
  --output_json "${RESULTS_DIR}/benchmark_results.json" \
  --output_v3 "${RESULTS_DIR}/benchmark_results_v3.json" \
  --model "${HF_MODEL}" \
  --quantization "${QUANT_NAME}" \
  --git_sha "${GIT_SHA}" \
  --workflow_run_id "${WORKFLOW_RUN_ID}" \
  --workflow_run_url "${WORKFLOW_RUN_URL}" \
  --gpu_name "$GPU_NAME" \
  --cuda_driver_version "$CUDA_DRIVER_VERSION"

# Save additional metadata (written via a serializer, never string-built).
BENCH_MODEL="${HF_MODEL}" BENCH_QUANT="${QUANT_NAME}" BENCH_NUM_RUNS="${NUM_RUNS}" \
  BENCH_RUNNER="${RUNNER}" BENCH_SHA="${GIT_SHA}" BENCH_RUN_ID="${WORKFLOW_RUN_ID}" \
  BENCH_RUN_URL="${WORKFLOW_RUN_URL}" BENCH_GPU="${GPU_NAME}" \
  RESULTS_DIR="${RESULTS_DIR}" python3 - <<'PY'
import datetime
import json
import os

metadata = {
    "model": os.environ["BENCH_MODEL"],
    "quantization": os.environ["BENCH_QUANT"],
    "num_runs": int(os.environ["BENCH_NUM_RUNS"]),
    "runner": os.environ["BENCH_RUNNER"],
    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    ),
    "git_sha": os.environ["BENCH_SHA"],
    "workflow_run_id": os.environ["BENCH_RUN_ID"],
    "workflow_run_url": os.environ["BENCH_RUN_URL"],
    "gpu_name": os.environ["BENCH_GPU"],
}
with open(os.path.join(os.environ.get("RESULTS_DIR", "."), "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
PY
echo "::endgroup::"
