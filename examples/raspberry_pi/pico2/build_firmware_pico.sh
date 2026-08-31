#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# build_firmware_pico.sh
# Simple script to cross-compile ExecuTorch and build Pico2 firmware with optional model input

set -euo pipefail

# Paths
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"  # examples/raspberry_pi/ -> root dir
PICO2_DIR="${ROOT_DIR}/examples/raspberry_pi/pico2"
BUILD_DIR="${PICO2_DIR}/build"
EXECUTORCH_BUILD_DIR="${ROOT_DIR}/cmake-out"

# Pico SDK 2.0's mbedtls requires this for CMake >= 3.30
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Portable nproc: use nproc on Linux, sysctl on macOS
if command -v nproc &>/dev/null; then
  NPROC=$(nproc)
else
  NPROC=$(sysctl -n hw.ncpu)
fi

# Source ARM toolchain if available and not already on PATH
if ! command -v arm-none-eabi-gcc &>/dev/null; then
  SETUP_PATH="${ROOT_DIR}/examples/arm/arm-scratch/setup_path.sh"
  if [ -f "${SETUP_PATH}" ]; then
    source "${SETUP_PATH}"
  else
    # Try to find the toolchain directly
    TOOLCHAIN_BIN=$(find "${ROOT_DIR}/examples/arm/arm-scratch" -name "arm-none-eabi-gcc" -type f 2>/dev/null | head -1)
    if [ -n "${TOOLCHAIN_BIN:-}" ]; then
      export PATH="$(dirname "${TOOLCHAIN_BIN}"):${PATH}"
    else
      echo "Error: arm-none-eabi-gcc not found. Run: ./examples/arm/setup.sh --i-agree-to-the-contained-eula"
      exit 1
    fi
  fi
fi

echo "Using ARM toolchain: $(which arm-none-eabi-gcc)"

# Default model
DEFAULT_MODEL="default_model.pte"

usage() {
  echo "Usage: $0 [--clean] [--cmsis] [--model=path/to/model.pte]"
  echo "  --clean           Clean build directories"
  echo "  --cmsis           Build with CMSIS-NN INT8 kernels (requires cortex_m backend)"
  echo "  --model=FILE      Specify model file to embed (relative to pico2/)"
  exit 1
}

# Parse args
MODEL_INPUT=""
CLEAN_BUILD=0
USE_CMSIS=0

for arg in "$@"; do
  case $arg in
    --clean)
      CLEAN_BUILD=1
      shift
      ;;
    --cmsis)
      USE_CMSIS=1
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

# Resolve the model path for selective build. Using EXECUTORCH_SELECT_OPS_MODEL
# auto-detects the exact operators the model needs from the .pte file, avoiding
# "Operator missing" errors at runtime.
# Note: skip selective build for CMSIS-NN models — their cortex_m:: ops are
# registered by the cortex_m backend, not by portable kernel codegen.
SELECT_OPS_FLAGS=""
if [ $USE_CMSIS -eq 0 ] && [ -n "$MODEL_INPUT" ] && [ -f "${PICO2_DIR}/${MODEL_INPUT}" ]; then
  MODEL_ABS_PATH="$(cd "${PICO2_DIR}" && realpath "${MODEL_INPUT}")"
  SELECT_OPS_FLAGS="-DEXECUTORCH_SELECT_OPS_MODEL=${MODEL_ABS_PATH}"
  echo "Using selective build from model: ${MODEL_ABS_PATH}"
fi

CMSIS_FLAGS=()
if [ $USE_CMSIS -eq 1 ]; then
  echo "CMSIS-NN mode: building with Cortex-M backend and CMSIS-NN kernels"
  CMSIS_FLAGS=(
    -DEXECUTORCH_BUILD_CORTEX_M=ON
  )
fi

cmake -B "${EXECUTORCH_BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ROOT_DIR}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DTARGET_CPU=cortex-m33+nofp \
  -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON \
  -DEXECUTORCH_PAL_DEFAULT=minimal \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DEXECUTORCH_ENABLE_LOGGING=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DCMAKE_INSTALL_PREFIX="${EXECUTORCH_BUILD_DIR}" \
  ${SELECT_OPS_FLAGS} \
  ${CMSIS_FLAGS[@]+"${CMSIS_FLAGS[@]}"} \
  "${ROOT_DIR}"

cmake --build "${EXECUTORCH_BUILD_DIR}" --target install -j${NPROC}

echo "ExecuTorch cross compile complete."

# Step 2: Build firmware for Pico2 with model input

cd "${PICO2_DIR}"

PICO_CMAKE_FLAGS=(-DPICO_BOARD=pico2 -DCMAKE_BUILD_TYPE=Release)

if [ $USE_CMSIS -eq 1 ]; then
  PICO_CMAKE_FLAGS+=(-DUSE_CMSIS_NN=ON)
fi

if [ -n "$MODEL_INPUT" ]; then
  # Use specified model
  if [ ! -f "${MODEL_INPUT}" ]; then
    echo "Error: Specified model file '${MODEL_INPUT}' not found in pico2 directory."
    exit 1
  fi
  echo "Building firmware with model: ${MODEL_INPUT}"
  PICO_CMAKE_FLAGS+=(-DINPUT_MODEL="./${MODEL_INPUT}")
else
  # Use default model
  echo "Building firmware with default model: ${DEFAULT_MODEL}"
  PICO_CMAKE_FLAGS+=(-DINPUT_MODEL="./${DEFAULT_MODEL}")
fi

cmake -B "${BUILD_DIR}" "${PICO_CMAKE_FLAGS[@]}"

cmake --build "${BUILD_DIR}" -j${NPROC}

echo "Firmware build complete. Output in ${BUILD_DIR}, Binary: executorch_pico.uf2"
