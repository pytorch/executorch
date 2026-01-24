#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
EXECUTORCH_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
BUILD_DIR="$SCRIPT_DIR/cmake-out"

# Resolve JAVA_HOME
if [ -z "$JAVA_HOME" ]; then
    JAVA_BIN=$(readlink -f $(which javac))
    export JAVA_HOME=$(dirname $(dirname $JAVA_BIN))
    echo "Detected JAVA_HOME: $JAVA_HOME"
fi

# 1. Build Native Library
echo "Building Native Library in $BUILD_DIR..."
mkdir -p "$BUILD_DIR"
pushd "$EXECUTORCH_ROOT"

# Use the 'jni' preset we added to CMakePresets.json
cmake --preset jni

# Build the targets
cmake --build cmake-out --target executorch_jni -j$(nproc)

popd

# Symlink libraries from the root cmake-out to local build dir for Java to find
# The preset typically builds in 'cmake-out' in the root
ROOT_BUILD_DIR="$EXECUTORCH_ROOT/cmake-out"
ln -sf "$ROOT_BUILD_DIR/extension/android/libexecutorch_jni.so" "$BUILD_DIR/libexecutorch.so"

# Finding fbjni in the root build tree can be tricky depending on how FetchContent put it.
# Usually it's in _deps/fbjni-build
FBJNI_LIB=$(find "$ROOT_BUILD_DIR" -name "libfbjni.so" | head -n 1)
ln -sf "$FBJNI_LIB" "$BUILD_DIR/libfbjni.so"

# 2. Compile FBJNI Java Sources
echo "Compiling FBJNI Java Sources..."
FBJNI_SRC_DIR="$ROOT_BUILD_DIR/_deps/fbjni-src/java"
if [ -d "$FBJNI_SRC_DIR" ]; then
    # Patch FBJNI sources to remove Nullable annotation usage since we don't have the dependency
    echo "Patching FBJNI sources..."
    find "$FBJNI_SRC_DIR" -name "*.java" -exec sed -i '/import javax.annotation.Nullable;/d' {} \;
    find "$FBJNI_SRC_DIR" -name "*.java" -exec sed -i 's/@Nullable//g' {} \;
    
    # Patch FBJNI to use System.loadLibrary instead of NativeLoader
    find "$FBJNI_SRC_DIR" -name "*.java" -exec sed -i '/import com.facebook.soloader.nativeloader.NativeLoader;/d' {} \;
    find "$FBJNI_SRC_DIR" -name "*.java" -exec sed -i 's/NativeLoader.loadLibrary/System.loadLibrary/g' {} \;

    find "$FBJNI_SRC_DIR" -name "*.java" > "$BUILD_DIR/fbjni_sources.txt"
    javac -d "$BUILD_DIR/classes" -cp "$BUILD_DIR/classes" @"$BUILD_DIR/fbjni_sources.txt"
else
    echo "Warning: FBJNI source directory not found at $FBJNI_SRC_DIR"
fi

# 3. Compile Executorch Java Sources
echo "Compiling Executorch Java Sources..."
ANDROID_JAVA_SRC="$EXECUTORCH_ROOT/extension/android/executorch_android/src/main/java"
# Find all java files
find "$ANDROID_JAVA_SRC" -name "*.java" > "$BUILD_DIR/sources.txt"
javac -d "$BUILD_DIR/classes" -cp "$BUILD_DIR/classes" @"$BUILD_DIR/sources.txt"

# 4. Compile Example
echo "Compiling Example..."
javac -d "$BUILD_DIR/classes" -cp "$BUILD_DIR/classes" "$SCRIPT_DIR/SimpleInference.java"

# 5. Run Example (if model provided)
if [ -n "$1" ]; then
    echo "Running Example..."
    # We need to set correct library path
    # libexecutorch_jni.so is in $BUILD_DIR/
    
    # Also need to find libc++_shared.so or similar if fbjni needs it? 
    # On linux usually standard shared libs work.
    
    java -cp "$BUILD_DIR/classes" \
         -Djava.library.path="$BUILD_DIR" \
         com.example.executorch.SimpleInference "$1"
else
    echo "Build success. To run: ./build_and_run_linux.sh <path_to_model.pte>"
fi
