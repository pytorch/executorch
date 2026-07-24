#!/bin/bash
# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Export a model and build firmware for the AXON NPU.
#
# Usage:
#   # Source NCS toolchain first
#   source ~/ncs-workspace/nrf-connect-sdk-env.sh
#
#   # Run with defaults (hello_axon sample)
#   ./backends/nordic/scripts/run.sh
#
#   # Or specify a custom export script
#   ./backends/nordic/scripts/run.sh \
#     --sample=examples/nordic/hello_axon \
#     --board=nrf54lm20dk/nrf54lm20b/cpuapp
#
# Environment variables:
#   SDK_EDGE_AI_PATH  - Path to Nordic sdk-edge-ai (required for AXON compilation)
#   PYTHON            - Python interpreter for model export (default: python3)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
ET_ROOT="$(cd "$BACKEND_DIR/../.." && pwd)"

# Defaults
SAMPLE="${ET_ROOT}/examples/nordic/hello_axon"
BOARD="nrf54lm20dk/nrf54lm20b/cpuapp"
BUILD_DIR=""
PYTHON="${PYTHON:-python3}"
SKIP_EXPORT=0
BUILD_ONLY=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample=*) SAMPLE="${1#*=}" ;;
        --board=*) BOARD="${1#*=}" ;;
        --build-dir=*) BUILD_DIR="${1#*=}" ;;
        --python=*) PYTHON="${1#*=}" ;;
        --skip-export) SKIP_EXPORT=1 ;;
        --build-only) BUILD_ONLY=1 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --sample=PATH       Sample directory (default: examples/nordic/hello_axon)"
            echo "  --board=BOARD       Zephyr board target (default: nrf54lm20dk/nrf54lm20b/cpuapp)"
            echo "  --build-dir=PATH    Build output directory (default: <sample>/build)"
            echo "  --python=PYTHON     Python for model export (default: python3)"
            echo "  --skip-export       Skip model export, use existing model_pte.h"
            echo "  --build-only        Build firmware but don't flash"
            echo ""
            echo "Environment:"
            echo "  SDK_EDGE_AI_PATH    Path to Nordic sdk-edge-ai (required)"
            echo "  Source nrf-connect-sdk-env.sh before running"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

SAMPLE="$(cd "$SAMPLE" && pwd)"
SAMPLE_NAME="$(basename "$SAMPLE")"
BUILD_DIR="${BUILD_DIR:-${SAMPLE}/build/${SAMPLE_NAME}}"

echo "=== Nordic AXON: Export and Build ==="
echo "  Sample:    $SAMPLE"
echo "  Board:     $BOARD"
echo "  Build dir: $BUILD_DIR"
echo "  SDK:       ${SDK_EDGE_AI_PATH:-NOT SET}"
echo ""

# ── Step 0: Validate environment ──────────────────────────────────
echo "--- Checking environment ---"
if ! bash "$SCRIPT_DIR/setup.sh" 2>/dev/null; then
    echo ""
    echo "WARNING: Environment check found issues (see above)."
    echo "  Continuing anyway — some steps may fail."
    echo ""
fi

# Check west is available (needed for firmware build)
if ! command -v west &>/dev/null; then
    echo "ERROR: 'west' not found."
    echo "  Source your NCS toolchain environment first:"
    echo "  source ~/ncs-workspace/nrf-connect-sdk-env.sh"
    exit 1
fi

# ── Step 1: Export model ──────────────────────────────────────────
if [ "$SKIP_EXPORT" -eq 0 ] && [ -f "$SAMPLE/export_model.py" ]; then
    echo ""
    echo "--- Exporting model ---"

    # Set up the export venv if it doesn't exist yet
    if [ ! -d "$SAMPLE/.venv" ]; then
        if [ -f "$SAMPLE/setup_export_env.sh" ]; then
            echo "First run — setting up export environment..."
            bash "$SAMPLE/setup_export_env.sh"
        fi
    fi

    # Export using uv (isolated from NCS Python).
    # Must unset PYTHONHOME/PYTHONPATH to avoid NCS toolchain conflicts.
    if command -v uv &>/dev/null && [ -d "$SAMPLE/.venv" ]; then
        PYTHONHOME= PYTHONPATH= SDK_EDGE_AI_PATH="${SDK_EDGE_AI_PATH}" \
            uv run --directory "$SAMPLE" python export_model.py
    else
        # Fallback: use whatever python is available
        SDK_EDGE_AI_PATH="${SDK_EDGE_AI_PATH}" \
            "$PYTHON" "$SAMPLE/export_model.py"
    fi
else
    if [ "$SKIP_EXPORT" -eq 0 ]; then
        echo "No export_model.py found in $SAMPLE — skipping export."
    else
        echo "Skipping export (--skip-export)."
    fi
fi

# Check model_pte.h exists
if [ ! -f "$SAMPLE/src/model_pte.h" ]; then
    echo "ERROR: $SAMPLE/src/model_pte.h not found."
    echo "  Run the export step first. If using uv:"
    echo "    cd $SAMPLE && ./setup_export_env.sh"
    echo "    PYTHONHOME= SDK_EDGE_AI_PATH=~/sdk-edge-ai uv run python export_model.py"
    exit 1
fi

# ── Step 2: Build firmware ────────────────────────────────────────
echo ""
echo "--- Building firmware ---"

EXTRA_MODULES="${ET_ROOT}"
if [ -n "$SDK_EDGE_AI_PATH" ] && [ -d "$SDK_EDGE_AI_PATH" ]; then
    EXTRA_MODULES="${ET_ROOT};${SDK_EDGE_AI_PATH}"
fi

west build \
    -b "$BOARD" \
    "$SAMPLE" \
    -d "$BUILD_DIR" \
    --no-sysbuild \
    -p \
    -- \
    -DZEPHYR_EXTRA_MODULES="$EXTRA_MODULES"

echo ""
echo "=== Build complete ==="
echo "  Hex:  $BUILD_DIR/zephyr/zephyr.hex"
echo "  ELF:  $BUILD_DIR/zephyr/zephyr.elf"

# ── Step 3: Flash ─────────────────────────────────────────────────
if [ "$BUILD_ONLY" -eq 0 ]; then
    echo ""
    echo "--- Flashing ---"
    west flash --build-dir "$BUILD_DIR"
    echo ""
    echo "Flash complete. Open serial console (115200 baud) to see output."
else
    echo ""
    echo "Build only — skipping flash."
    echo "  Flash with: west flash --build-dir $BUILD_DIR"
fi
