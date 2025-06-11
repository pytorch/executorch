#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
SOURCE_DIR=$(realpath "$(dirname "$0")/../../..")

# Check if buck2 exists
BUCK_PATH=${BUCK2:-buck2}
if [ -z "$BUCK2" ]; then
    echo "Info: BUCK2 environment variable is not set." >&2
fi

# Check if the ANDROID_NDK environment variable is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable is not set." >&2
    exit 1
fi

# Create and enter the build directory
# Set build directory
build_dir="cmake-android-out"
cd "$SOURCE_DIR"
rm -rf "${build_dir}"

# Configure the project with CMake
# Note: Add any additional configuration options you need here
cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
      -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_NATIVE_API_LEVEL=26 \
      -DANDROID_PLATFORM=android-26 \
      -DEXECUTORCH_BUILD_NEURON=ON \
      -B"${build_dir}"

# Build the project
cmake --build "${build_dir}" --target install --config Release -j5

# Switch back to the original directory
cd - > /dev/null

# Print a success message
echo "Build successfully completed."
