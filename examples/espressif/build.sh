#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build script for the ExecuTorch ESP32 executor runner example.
#
# Prerequisites:
#   - ESP-IDF v5.1+ installed and sourced (. $IDF_PATH/export.sh)
#   - ExecuTorch cross-compiled for the ESP32 target
#   - Python 3.8+
#
# Usage:
#   ./build.sh [--target esp32|esp32s3] [--pte <model.pte>] [--clean]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/project"
TARGET="esp32s3"
PTE_FILE=""
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --pte)
            PTE_FILE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--target esp32|esp32s3] [--pte <model.pte>] [--clean]"
            echo ""
            echo "Options:"
            echo "  --target    ESP32 target chip (default: esp32s3)"
            echo "  --pte       Path to the .pte model file to embed"
            echo "  --clean     Clean build directory before building"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
if [ -z "${IDF_PATH:-}" ]; then
    echo "ERROR: IDF_PATH is not set. Please source ESP-IDF:"
    echo "  . \$IDF_PATH/export.sh"
    exit 1
fi

echo "=== ExecuTorch ESP32 Executor Runner Build ==="
echo "Target: ${TARGET}"
echo "ExecuTorch root: ${ET_ROOT}"
echo "ESP-IDF: ${IDF_PATH}"

# Convert PTE to header if provided
if [ -n "${PTE_FILE}" ]; then
    if [ ! -f "${PTE_FILE}" ]; then
        echo "ERROR: PTE file not found: ${PTE_FILE}"
        exit 1
    fi

    echo "Converting PTE to header: ${PTE_FILE}"
    HEADER_DIR="${PROJECT_DIR}/build/model"
    mkdir -p "${HEADER_DIR}"
    python3 "${SCRIPT_DIR}/executor_runner/pte_to_header.py" \
        --pte "${PTE_FILE}" \
        --outdir "${HEADER_DIR}"
    echo "Model header generated: ${HEADER_DIR}/model_pte.h"
fi

# Navigate to project directory
cd "${PROJECT_DIR}"

# Clean if requested
if [ "${CLEAN}" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build sdkconfig
fi

# Set target
echo "Setting target to ${TARGET}..."
idf.py set-target "${TARGET}"

# Build
echo "Building..."
idf.py build

echo ""
echo "=== Build complete ==="
echo ""
echo "To flash and monitor:"
echo "  cd ${PROJECT_DIR}"
echo "  idf.py -p /dev/ttyUSB0 flash monitor"
echo ""
echo "To just monitor:"
echo "  idf.py -p /dev/ttyUSB0 monitor"
