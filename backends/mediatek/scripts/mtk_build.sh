#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
EXECUTORCH_ROOT=$(realpath "$(dirname "$0")/../../..")
echo EXECUTORCH_ROOT=${EXECUTORCH_ROOT}

# Check if the ANDROID_NDK environment variable is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable is not set." >&2
    exit 1
fi

# Enter the build directory
cd "$EXECUTORCH_ROOT"

# Set build directory
build_dir="cmake-android-out"
rm -rf "${build_dir}"

# Configure the project with CMake
# Note: Add any additional configuration options you need here
cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
      -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_NATIVE_API_LEVEL=26 \
      -DANDROID_PLATFORM=android-26 \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
      -DEXECUTORCH_BUILD_NEURON=ON \
      -B"${build_dir}"

# Build the project
cmake --build "${build_dir}" --target install --config Release -j5

# Switch back to the original directory
cd - > /dev/null

# Print a success message
echo "Build successfully completed."
