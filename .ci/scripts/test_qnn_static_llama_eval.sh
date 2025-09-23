#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# Download QNN_SDK. If already downloaded, export environment path
source "$(dirname "${BASH_SOURCE[0]}")/../../backends/qualcomm/scripts/install_qnn_sdk.sh"
install_qnn

export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang"
export PYTHONPATH=".."
cp schema/program.fbs exir/_serialize/program.fbs
cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
cp -f build-x86/backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so backends/qualcomm/python
cp -f build-x86/backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so backends/qualcomm/python

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"

# -------------------------------
# Parse args
# -------------------------------
EXTRA_FLAGS=""
THRESHOLD=62.0  # default fallback

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flags)
      EXTRA_FLAGS="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Config
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
MODEL="qwen2_5-0_5b"
MAX_SEQ=1024
PTQ="16a4w"
THRESHOLD=62.0   # regression guardrail

EXTRA_FLAGS="$@"

# Run command and capture *both stdout and stderr*
LOG_FILE="eval_${MODEL}_$(date +%Y%m%d_%H%M%S).log"

echo ">>> Running evaluation with flags: $EXTRA_FLAGS | threshold: $THRESHOLD"
$PYTHON_EXECUTABLE -m executorch.examples.qualcomm.oss_scripts.llama.eval_llama_qnn \
  --decoder_model "$MODEL" \
  --quant_linear_only \
  --max_seq_length "$MAX_SEQ" \
  --ptq "$PTQ" \
  $EXTRA_FLAGS 2>&1 | tee "$LOG_FILE"

# Extract last word_perplexity
LAST_PERP=$(grep "INFO:root:wikitext:" "$LOG_FILE" | tail -n 1 | sed -E "s/.*'word_perplexity,none': ([0-9.]+).*/\1/")

if [[ -z "$LAST_PERP" ]]; then
  echo "❌ Could not find word_perplexity in logs!"
  exit 1
fi

echo ">>> Last word_perplexity = $LAST_PERP"

# Compare against threshold
awk -v val="$LAST_PERP" -v thr="$THRESHOLD" 'BEGIN {exit (val > thr)}'
if [[ $? -ne 0 ]]; then
  echo "❌ Regression detected: word_perplexity ($LAST_PERP) > threshold ($THRESHOLD)"
  exit 1
fi

echo "✅ Check passed: word_perplexity ($LAST_PERP) <= $THRESHOLD"
