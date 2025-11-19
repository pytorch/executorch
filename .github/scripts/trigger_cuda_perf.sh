#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quick script to trigger cuda-perf workflow via GitHub CLI
# Usage:
#   ./trigger_cuda_perf.sh                                          # Use defaults
#   ./trigger_cuda_perf.sh "openai/whisper-medium"                  # Single model
#   ./trigger_cuda_perf.sh "openai/whisper-small,google/gemma-3-4b-it" "non-quantized,quantized-int4-tile-packed" "100"

set -e

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

# Default parameters
MODELS="${1:-openai/whisper-small}"
QUANT="${2:-non-quantized}"
NUM_RUNS="${3:-50}"
RANDOM_MODEL="${4:-false}"

# Display configuration
echo "========================================="
echo "Triggering cuda-perf workflow"
echo "========================================="
echo "Models:         $MODELS"
echo "Quantizations:  $QUANT"
echo "Num runs:       $NUM_RUNS"
echo "Random model:   $RANDOM_MODEL"
echo "========================================="
echo ""

# Trigger workflow
# Use -R to specify repository since we might not be in a git repo
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
