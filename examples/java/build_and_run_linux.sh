#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
EXECUTORCH_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
BUILD_DIR="$SCRIPT_DIR/cmake-out"
CMAKE_OUT="$EXECUTORCH_ROOT/cmake-out"

# Activate conda environment with torch
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate executorch
    echo "Activated conda environment: executorch"
fi

# Resolve JAVA_HOME
if [ -z "$JAVA_HOME" ]; then
    JAVA_BIN=$(readlink -f $(which javac))
    export JAVA_HOME=$(dirname $(dirname $JAVA_BIN))
    echo "Detected JAVA_HOME: $JAVA_HOME"
fi

# Set PYTHON_EXECUTABLE to ensure cmake uses the right python
export PYTHON_EXECUTABLE=$(which python)
echo "Using Python: $PYTHON_EXECUTABLE"

# 1. Build Native Library
echo "Building Native Library..."
mkdir -p "$BUILD_DIR"
pushd "$EXECUTORCH_ROOT"

# Clean cmake cache if it exists to avoid conflicts
if [ -f "$CMAKE_OUT/CMakeCache.txt" ]; then
    echo "Cleaning previous cmake cache..."
    rm -rf "$CMAKE_OUT"
fi

# Configure without optimized kernels (uses XNNPACK for linear ops instead)
# The custom LLM ops require optimized kernels, so we skip those and rely on XNNPACK
cmake . -B"$CMAKE_OUT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-16 \
    -DCMAKE_CXX_COMPILER=clang++-16 \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DJAVA_HOME="$JAVA_HOME" \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DEXECUTORCH_BUILD_HOST_JAVA=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXAMPLES=OFF \
    -DEXECUTORCH_BUILD_DEVTOOLS=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON

# Build the targets
cmake --build "$CMAKE_OUT" --target executorch_jni -j$(nproc)

popd

# Symlink libraries from the root cmake-out to local build dir for Java to find
ln -sf "$CMAKE_OUT/extension/android/libexecutorch_jni.so" "$BUILD_DIR/libexecutorch.so"

# 3. Compile Executorch Java Sources
echo "Compiling Executorch Java Sources..."
ANDROID_JAVA_SRC="$EXECUTORCH_ROOT/extension/android/executorch_android/src/main/java"
mkdir -p "$BUILD_DIR/classes"
# Find all java files
find "$ANDROID_JAVA_SRC" -name "*.java" > "$BUILD_DIR/sources.txt"
javac -d "$BUILD_DIR/classes" -cp "$BUILD_DIR/classes" @"$BUILD_DIR/sources.txt"

# 4. Compile Example
echo "Compiling Example..."
javac -d "$BUILD_DIR/classes" -cp "$BUILD_DIR/classes" "$SCRIPT_DIR/SimpleInference.java"
javac -d "$BUILD_DIR/classes" -cp "$BUILD_DIR/classes" "$SCRIPT_DIR/LlamaChat.java"

# 5. Run Example (if model provided)
if [ -n "$1" ]; then
    echo "Running Example..."
    java -cp "$BUILD_DIR/classes" \
         -Djava.library.path="$BUILD_DIR" \
         com.example.executorch.SimpleInference "$1"
else
    echo "Build success. To run: ./build_and_run_linux.sh <path_to_model.pte>"
fi
