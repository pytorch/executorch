#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Simple build script for the multimodal example

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="${SCRIPT_DIR}/../../.."

echo "Building multimodal example..."
echo "ExecuteTorch root: $EXECUTORCH_ROOT"

# Create build directory
BUILD_DIR="${SCRIPT_DIR}/build_example"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# Configure with CMake
cmake -DEXECUTORCH_ROOT="$EXECUTORCH_ROOT" \
      -DCMAKE_PREFIX_PATH="$EXECUTORCH_ROOT/cmake-out" \
      -DCMAKE_BUILD_TYPE=Release \
      -S "$SCRIPT_DIR" \
      -B . \
      -f "$SCRIPT_DIR/multimodal_example_build.cmake"

# Build
cmake --build . --parallel

echo "Build complete!"
echo "Executable: $BUILD_DIR/multimodal_example"
echo ""
echo "Usage example:"
echo "  $BUILD_DIR/multimodal_example model.pte tokenizer.json --text 'Transcribe this audio:' --audio audio.pt"