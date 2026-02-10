#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build script for ExecuTorch Java package (desktop platforms)
#
# This script builds:
# 1. The native JNI library (libexecutorch_jni.so/dylib/dll)
# 2. The Java JAR file
#
# Usage:
#   ./build_java.sh [options]
#
# Options:
#   --cmake-only     Only build native libraries with CMake
#   --jar-only       Only build the Java JAR (assumes native libs exist)
#   --clean          Clean build directories before building
#   --help           Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CMAKE_BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out-java"
NATIVE_LIB_DIR="${SCRIPT_DIR}/build/native"

# Detect JAVA_HOME if not set
detect_java_home() {
    if [ -n "${JAVA_HOME}" ]; then
        echo "Using JAVA_HOME: ${JAVA_HOME}"
        return
    fi

    # Try to detect Java installation
    local java_path
    if command -v java &> /dev/null; then
        java_path=$(command -v java)
        # Follow symlinks to find actual Java installation
        while [ -L "$java_path" ]; do
            java_path=$(readlink "$java_path")
        done

        # Go up from bin/java to the Java home
        local java_bin_dir=$(dirname "$java_path")
        local potential_home=$(dirname "$java_bin_dir")

        # Check if this looks like a valid JAVA_HOME
        if [ -f "${potential_home}/include/jni.h" ]; then
            export JAVA_HOME="${potential_home}"
            echo "Detected JAVA_HOME: ${JAVA_HOME}"
            return
        fi
    fi

    # macOS: Try /usr/libexec/java_home
    if [ "$(uname -s)" = "Darwin" ] && command -v /usr/libexec/java_home &> /dev/null; then
        export JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null || true)
        if [ -n "${JAVA_HOME}" ] && [ -f "${JAVA_HOME}/include/jni.h" ]; then
            echo "Detected JAVA_HOME (macOS): ${JAVA_HOME}"
            return
        fi
    fi

    # Linux: Common locations
    for dir in /usr/lib/jvm/java-* /usr/lib/jvm/default-java /usr/java/latest; do
        if [ -d "$dir" ] && [ -f "${dir}/include/jni.h" ]; then
            export JAVA_HOME="$dir"
            echo "Detected JAVA_HOME: ${JAVA_HOME}"
            return
        fi
    done

    echo "Warning: JAVA_HOME not set and could not be auto-detected."
    echo "Please set JAVA_HOME environment variable to your JDK installation."
    echo "Example: export JAVA_HOME=/usr/lib/jvm/java-11-openjdk"
}

detect_java_home

# Determine OS and architecture
get_os_arch() {
    local os arch

    case "$(uname -s)" in
        Darwin*)
            os="darwin"
            ;;
        Linux*)
            os="linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            os="windows"
            ;;
        *)
            os="linux"
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="aarch64"
            ;;
        *)
            arch="$(uname -m)"
            ;;
    esac

    echo "${os}-${arch}"
}

OS_ARCH=$(get_os_arch)

# Number of parallel jobs
if command -v nproc &> /dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &> /dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

# Parse arguments
CMAKE_ONLY=false
JAR_ONLY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cmake-only)
            CMAKE_ONLY=true
            shift
            ;;
        --jar-only)
            JAR_ONLY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            head -25 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directories..."
    rm -rf "${CMAKE_BUILD_DIR}"
    rm -rf "${SCRIPT_DIR}/build"
fi

# Build native libraries with CMake
build_native() {
    echo "Building native JNI libraries..."
    echo "  OS/Arch: ${OS_ARCH}"
    echo "  CMake build dir: ${CMAKE_BUILD_DIR}"
    echo "  Parallel jobs: ${NPROC}"

    mkdir -p "${CMAKE_BUILD_DIR}"
    cd "${CMAKE_BUILD_DIR}"

    # Configure and build ExecutorTorch with Java JNI extension
    # Using a single CMake configuration that includes the Java extension
    cmake "${EXECUTORCH_ROOT}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DEXECUTORCH_BUILD_JAVA_JNI=ON \
        "$@"

    # Build all targets including executorch_jni
    cmake --build . -j"${NPROC}"

    # Copy native library to output directory
    mkdir -p "${NATIVE_LIB_DIR}/${OS_ARCH}"

    case "$(uname -s)" in
        Darwin*)
            find "${CMAKE_BUILD_DIR}" -name "libexecutorch_jni.dylib" -exec cp {} "${NATIVE_LIB_DIR}/${OS_ARCH}/" \;
            ;;
        Linux*)
            find "${CMAKE_BUILD_DIR}" -name "libexecutorch_jni.so" -exec cp {} "${NATIVE_LIB_DIR}/${OS_ARCH}/" \;
            ;;
        CYGWIN*|MINGW*|MSYS*)
            find "${CMAKE_BUILD_DIR}" -name "executorch_jni.dll" -exec cp {} "${NATIVE_LIB_DIR}/${OS_ARCH}/" \;
            ;;
    esac

    echo "Native library built: ${NATIVE_LIB_DIR}/${OS_ARCH}/"
    ls -la "${NATIVE_LIB_DIR}/${OS_ARCH}/"
}

# Build Java JAR
build_jar() {
    echo "Building Java JAR..."
    cd "${SCRIPT_DIR}"

    if command -v ./gradlew &> /dev/null; then
        ./gradlew build
    elif command -v gradle &> /dev/null; then
        gradle build
    else
        echo "Error: Gradle not found. Please install Gradle or use the Gradle wrapper."
        exit 1
    fi

    echo "JAR built successfully!"
    ls -la "${SCRIPT_DIR}/build/libs/"
}

# Main
if [ "$JAR_ONLY" = true ]; then
    build_jar
elif [ "$CMAKE_ONLY" = true ]; then
    build_native
else
    build_native
    build_jar
fi

echo ""
echo "Build complete!"
echo ""
echo "To use the library:"
echo "  1. Add ${SCRIPT_DIR}/build/libs/executorch-java-*.jar to your classpath"
echo "  2. Either:"
echo "     a. Set java.library.path to ${NATIVE_LIB_DIR}/${OS_ARCH}"
echo "     b. Or the JAR will auto-extract native libs at runtime"
