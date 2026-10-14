#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quick script to trigger the CUDA workflow (accuracy + benchmark, merged
# from the retired cuda-perf.yml) via GitHub CLI.
# Benchmarks reuse each accuracy cell's exported .pte/.ptd in place; the
# models/quantizations filters below only narrow the benchmark stage.
# Usage:
#   ./trigger_cuda_benchmark.sh                                          # Use defaults (all allowlisted pairs)
#   ./trigger_cuda_benchmark.sh "openai/whisper-medium"                  # Single model
#   ./trigger_cuda_benchmark.sh "openai/whisper-small,google/gemma-3-4b-it" "non-quantized,quantized-int4-tile-packed" "100"
#
# NOTE: perf allowlist lives in .ci/scripts/cuda_benchmark_from_e2e.sh and the
# test-model-cuda-e2e matrix in .github/workflows/cuda.yml. Models outside the
# accuracy matrix cannot be benchmarked via this script.

set -e

# Models covered by both accuracy e2e and perf tracking (see
# cuda_benchmark_from_e2e.sh is_bench_pair). whisper-medium is the one
# perf-first model backfilled into the accuracy matrix.
ALL_MODELS="mistralai/Voxtral-Mini-3B-2507,openai/whisper-small,openai/whisper-medium,openai/whisper-large-v3-turbo,google/gemma-3-4b-it,nvidia/parakeet-tdt,SocialLocalMobile/Qwen3.5-35B-A3B-HQQ-INT4"
ALL_QUANTIZATIONS="non-quantized,quantized-int4-tile-packed,quantized-int4-weight-only"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    echo ""
    echo "Quick install:"
    echo "  macOS:   brew install gh"
    echo "  Linux:   See https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    exit 1
fi

MODELS="${1:-}"
QUANT="${2:-}"
NUM_RUNS="${3:-50}"

# Display configuration
echo "========================================="
echo "Triggering cuda workflow (accuracy + benchmark)"
echo "========================================="
if [ -z "$MODELS" ]; then
    echo "Models:         (all allowlisted: $ALL_MODELS)"
else
    echo "Models:         $MODELS"
fi
if [ -z "$QUANT" ]; then
    echo "Quantizations:  (all allowlisted: $ALL_QUANTIZATIONS)"
else
    echo "Quantizations:  $QUANT"
fi
echo "Num runs:       $NUM_RUNS"
echo "========================================="

echo ""

# Trigger workflow (dispatch inputs only narrow the benchmark stage)
gh workflow run cuda.yml \
  -R pytorch/executorch \
  -f models="$MODELS" \
  -f quantizations="$QUANT" \
  -f num_runs="$NUM_RUNS"

if [ $? -eq 0 ]; then
    echo "✓ Workflow triggered successfully!"
    echo ""
    echo "View status:"
    echo "  gh run list --workflow=cuda.yml"
    echo ""
    echo "Watch the latest run:"
    echo "  gh run watch \$(gh run list --workflow=cuda.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
else
    echo "✗ Failed to trigger workflow"
    exit 1
fi
