#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quick script to trigger cuda-perf workflow via GitHub CLI
# Usage:
#   ./trigger_cuda_perf.sh                                          # Use defaults (random model + quant)
#   ./trigger_cuda_perf.sh --all                                     # Run ALL models with ALL quantizations
#   ./trigger_cuda_perf.sh "openai/whisper-medium"                  # Single model
#   ./trigger_cuda_perf.sh "openai/whisper-small,google/gemma-3-4b-it" "non-quantized,quantized-int4-tile-packed" "100"

set -e

# All available models and quantizations
ALL_MODELS="mistralai/Voxtral-Mini-3B-2507,openai/whisper-small,openai/whisper-medium,openai/whisper-large-v3-turbo,google/gemma-3-4b-it"
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

# Check for --all flag
RUN_ALL=false
if [ "${1:-}" = "--all" ] || [ "${1:-}" = "-a" ]; then
    RUN_ALL=true
    shift  # Remove the flag from arguments
fi

# Default parameters
if [ "$RUN_ALL" = true ]; then
    MODELS="$ALL_MODELS"
    QUANT="$ALL_QUANTIZATIONS"
    NUM_RUNS="${1:-50}"
    RANDOM_MODEL="false"
    echo "========================================="
    echo "Triggering cuda-perf workflow"
    echo "Mode: RUN ALL MODELS AND QUANTIZATIONS"
    echo "========================================="
    echo "Models:         ALL (5 models)"
    echo "Quantizations:  ALL (3 quantizations)"
    echo "Total configs:  15 combinations"
    echo "Num runs:       $NUM_RUNS"
    echo "========================================="
else
    MODELS="${1:-}"
    QUANT="${2:-}"
    NUM_RUNS="${3:-50}"
    RANDOM_MODEL="${4:-false}"

    # Display configuration
    echo "========================================="
    echo "Triggering cuda-perf workflow"
    echo "========================================="
    if [ -z "$MODELS" ]; then
        echo "Models:         (random selection)"
    else
        echo "Models:         $MODELS"
    fi
    if [ -z "$QUANT" ]; then
        echo "Quantizations:  (random selection)"
    else
        echo "Quantizations:  $QUANT"
    fi
    echo "Num runs:       $NUM_RUNS"
    echo "Random model:   $RANDOM_MODEL"
    echo "========================================="
fi

echo ""

# Trigger workflow
gh workflow run cuda-perf.yml \
  -R pytorch/executorch \
  -f models="$MODELS" \
  -f quantizations="$QUANT" \
  -f num_runs="$NUM_RUNS" \
  -f random_model="$RANDOM_MODEL"

if [ $? -eq 0 ]; then
    echo "✓ Workflow triggered successfully!"
    echo ""
    echo "View status:"
    echo "  gh run list --workflow=cuda-perf.yml"
    echo ""
    echo "Watch the latest run:"
    echo "  gh run watch \$(gh run list --workflow=cuda-perf.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
else
    echo "✗ Failed to trigger workflow"
    exit 1
fi
