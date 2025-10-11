#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
# build_firmware_pico.sh
# Simple script to cross-compile ExecuTorch and build Pico2 firmware with optional model input

set -e

# Paths
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"  # examples/rpi/ -> root dir
PICO2_DIR="${ROOT_DIR}/examples/rpi/pico2"
BUILD_DIR="${PICO2_DIR}/build"
EXECUTORCH_BUILD_DIR="${ROOT_DIR}/cmake-out"

# Default model
DEFAULT_MODEL="default_model.pte"

usage() {
  echo "Usage: $0 [--clean] [--model=path/to/model.pte]"
  echo "  --clean           Clean build directories"
  echo "  --model=FILE      Specify model file to embed (relative to pico2/)"
  exit 1
}

# Parse args
MODEL_INPUT=""
CLEAN_BUILD=0

for arg in "$@"; do
  case $arg in
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --model=*)
      MODEL_INPUT="${arg#*=}"
      shift
      ;;
    *)
      usage
      ;;
  esac
done

# Clean if requested
if [ $CLEAN_BUILD -eq 1 ]; then
  echo "Cleaning build directories..."
  rm -rf "${EXECUTORCH_BUILD_DIR}" "${BUILD_DIR}"
  echo "Clean complete."
  exit 0
fi

# Step 1: Cross compile ExecuTorch from root dir
echo "Cross compiling ExecuTorch baremetal ARM..."

cmake -B "${EXECUTORCH_BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT_DIR}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DTARGET_CPU=cortex-m0plus \
  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON \
  -DEXECUTORCH_PAL_DEFAULT=minimal \
  -DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DEXECUTORCH_ENABLE_LOGGING=OFF \
  -DEXECUTORCH_SELECT_ALL_OPS=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DCMAKE_INSTALL_PREFIX="${EXECUTORCH_BUILD_DIR}" \
  "${ROOT_DIR}"

cmake --build "${EXECUTORCH_BUILD_DIR}" --target install -j$(nproc)

echo "ExecuTorch cross compile complete."

# Step 2: Build firmware for Pico2 with model input

cd "${PICO2_DIR}"

if [ -n "$MODEL_INPUT" ]; then
  # Use specified model
  if [ ! -f "${MODEL_INPUT}" ]; then
    echo "Error: Specified model file '${MODEL_INPUT}' not found in pico2 directory."
    exit 1
  fi
  echo "Building firmware with model: ${MODEL_INPUT}"
  cmake -B "${BUILD_DIR}" -DPICO_BOARD=pico2 -DINPUT_MODEL="./${MODEL_INPUT}" -DCMAKE_BUILD_TYPE=Release
else
  # Use default model
  echo "Building firmware with default model: ${DEFAULT_MODEL}"
  cmake -B "${BUILD_DIR}" -DPICO_BOARD=pico2 -DINPUT_MODEL="./${DEFAULT_MODEL}" -DCMAKE_BUILD_TYPE=Release
fi

cmake --build "${BUILD_DIR}" -j$(nproc)

echo "Firmware build complete. Output in ${BUILD_DIR}, Binary: executorch_pico.uf2"
