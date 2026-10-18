#!/usr/bin/env bash
# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

# Cross-compile the ExecuTorch runtime, portable kernels and quantized kernels
# for an Espressif chip with the ESP-IDF toolchain.
#
# Run inside an activated ESP-IDF shell (. $IDF_PATH/export.sh):
#   examples/espressif/scripts/build_executorch_esp.sh [esp32s3|esp32|esp32s2|esp32p4|...]
#
# Environment:
#   EXECUTORCH_ROOT        ExecuTorch source tree (default: this repository)
#   ET_BUILD_DIR           output directory (default: $EXECUTORCH_ROOT/cmake-out-esp-<target>)
#   CMAKE                  CMake >= 3.24 (ExecuTorch requirement)
#   PYTHON_EXECUTABLE      Python with executorch and torch installed (kernel registration codegen)
#   ET_SELECT_OPS_LIST     comma separated ops for a selective build, e.g. "aten::add.out,..."
#   ET_SELECT_OPS_MODEL    a .pte whose operators define the selective build
set -euo pipefail

TARGET=${1:-${IDF_TARGET:-esp32s3}}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXECUTORCH_ROOT=${EXECUTORCH_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
BUILD_DIR=${ET_BUILD_DIR:-$EXECUTORCH_ROOT/cmake-out-esp-$TARGET}
CMAKE=${CMAKE:-cmake}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}

if [ -z "${IDF_PATH:-}" ]; then
    echo "build_executorch_esp: IDF_PATH is not set; run . \$IDF_PATH/export.sh first" >&2
    exit 1
fi

case "$TARGET" in
    esp32|esp32s2|esp32s3)
        TARGET_FLAGS="-mlongcalls -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-bzero -mdisable-hardware-atomics" ;;
    *)
        TARGET_FLAGS="-fno-builtin-memcpy -fno-builtin-memset -fno-builtin-bzero" ;;
esac

SELECT_ARGS=()
if [ -n "${ET_SELECT_OPS_LIST:-}" ]; then
    SELECT_ARGS+=(-DEXECUTORCH_SELECT_OPS_LIST="$ET_SELECT_OPS_LIST")
fi
if [ -n "${ET_SELECT_OPS_MODEL:-}" ]; then
    SELECT_ARGS+=(-DEXECUTORCH_SELECT_OPS_MODEL="$ET_SELECT_OPS_MODEL")
fi
if [ ${#SELECT_ARGS[@]} -eq 0 ]; then
    SELECT_ARGS+=(-DEXECUTORCH_BUILD_PORTABLE_OPS=ON)
fi

mkdir -p "$BUILD_DIR"
cd "$EXECUTORCH_ROOT"
"$CMAKE" --preset esp-baremetal -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$IDF_PATH/tools/cmake/toolchain-$TARGET.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=OFF \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=OFF \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
    -DEXECUTORCH_BUILD_CPUINFO=OFF \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
    -DEXECUTORCH_PAL_DEFAULT=minimal \
    "${SELECT_ARGS[@]}"

# Outside an ESP-IDF project the toolchain file leaves its flag response
# files empty; ESP32/S2/S3 code needs -mlongcalls to link.
for flags_file in cflags cxxflags asmflags; do
    printf '%s\n' $TARGET_FLAGS > "$BUILD_DIR/toolchain/$flags_file"
done
"$CMAKE" --build "$BUILD_DIR" -j"$(nproc)" --target install
echo "build_executorch_esp: libraries in $BUILD_DIR/lib, headers in $BUILD_DIR/include"
