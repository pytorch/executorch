#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script checks if the PyTorch commit hash in pytorch.txt matches
# the commit hash for the NIGHTLY_VERSION specified in torch_pin.py.
#
# It verifies by querying GitHub API to get the commit date and comparing
# it with the expected nightly date.

set -eu

# Get the directory of this script and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TORCH_PIN_FILE="$REPO_ROOT/torch_pin.py"
PYTORCH_TXT_FILE="$REPO_ROOT/.ci/docker/ci_commit_pins/pytorch.txt"

echo "================================================================================"
echo "Checking PyTorch commit pin consistency"
echo "================================================================================"
echo ""

# Check if required files exist
if [ ! -f "$TORCH_PIN_FILE" ]; then
    echo "Error: $TORCH_PIN_FILE not found"
    exit 1
fi

if [ ! -f "$PYTORCH_TXT_FILE" ]; then
    echo "Error: $PYTORCH_TXT_FILE not found"
    exit 1
fi

# Extract NIGHTLY_VERSION from torch_pin.py
NIGHTLY_VERSION=$(grep -oP 'NIGHTLY_VERSION\s*=\s*["\047]\K[^"\047]+' "$TORCH_PIN_FILE" || true)
if [ -z "$NIGHTLY_VERSION" ]; then
    echo "Error: Could not find NIGHTLY_VERSION in $TORCH_PIN_FILE"
    exit 1
fi

# Extract commit hash from pytorch.txt
PYTORCH_COMMIT=$(cat "$PYTORCH_TXT_FILE" | tr -d '[:space:]')
if [ -z "$PYTORCH_COMMIT" ]; then
    echo "Error: $PYTORCH_TXT_FILE is empty"
    exit 1
fi

# Extract date from NIGHTLY_VERSION (format: devYYYYMMDD)
if [[ ! "$NIGHTLY_VERSION" =~ dev([0-9]{8}) ]]; then
    echo "Error: Invalid nightly version format: $NIGHTLY_VERSION"
    echo "Expected format: devYYYYMMDD (e.g., dev20251004)"
    exit 1
fi

DATE_STR="${BASH_REMATCH[1]}"
YEAR="${DATE_STR:0:4}"
MONTH="${DATE_STR:4:2}"
DAY="${DATE_STR:6:2}"
EXPECTED_DATE="$YEAR-$MONTH-$DAY"

echo "Nightly version: $NIGHTLY_VERSION"
echo "Expected date: $EXPECTED_DATE"
echo "Commit hash: $PYTORCH_COMMIT"
echo ""

# Query GitHub API to get commit information
GITHUB_API_URL="https://api.github.com/repos/pytorch/pytorch/commits/$PYTORCH_COMMIT"
echo "Querying GitHub API: $GITHUB_API_URL"

# Use curl to get commit data
RESPONSE=$(curl -s -H "Accept: application/vnd.github.v3+json" \
                -H "User-Agent: Mozilla/5.0" \
                "$GITHUB_API_URL")

# Check if curl was successful
if [ $? -ne 0 ]; then
    echo "✗ Failed to query GitHub API"
    exit 1
fi

# Check if response contains error
if echo "$RESPONSE" | grep -q '"message".*"Not Found"'; then
    echo "✗ Commit not found on GitHub (404)"
    exit 1
fi

# Extract commit date using grep and sed (avoiding jq dependency)
# The commit date is in format: "date": "2025-10-13T07:00:00Z"
COMMIT_DATE=$(echo "$RESPONSE" | grep -oP '"committer"[^}]*"date":\s*"\K[^"T]+' | head -1)

if [ -z "$COMMIT_DATE" ]; then
    echo "✗ Could not extract commit date from GitHub API response"
    exit 1
fi

echo "Commit date from GitHub: $COMMIT_DATE"
echo ""

# Compare dates
echo "================================================================================"
echo "Verification Result"
echo "================================================================================"

if [ "$COMMIT_DATE" = "$EXPECTED_DATE" ]; then
    echo "✓ SUCCESS: PyTorch commit pin matches the nightly version!"
    echo ""
    echo "Commit $PYTORCH_COMMIT corresponds to $NIGHTLY_VERSION"
    echo ""
    echo "Reference: https://hud.pytorch.org/pytorch/pytorch/commit/$PYTORCH_COMMIT"
    exit 0
else
    echo "✗ ERROR: PyTorch commit pin does NOT match the nightly version!"
    echo ""
    echo "  Expected date: $EXPECTED_DATE"
    echo "  Actual date:   $COMMIT_DATE"
    echo ""
    echo "The commit in $PYTORCH_TXT_FILE"
    echo "does not correspond to NIGHTLY_VERSION=$NIGHTLY_VERSION"
    echo ""
    echo "Please verify and update the commit hash:"
    echo "1. Visit https://hud.pytorch.org/hud/pytorch/pytorch/nightly/"
    echo "2. Find the nightly build for $NIGHTLY_VERSION"
    echo "3. Copy the correct commit hash"
    echo "4. Update $PYTORCH_TXT_FILE"
    echo ""
    echo "Reference: https://hud.pytorch.org/pytorch/pytorch/commit/$PYTORCH_COMMIT"
    exit 1
fi
