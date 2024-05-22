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

# Check if the NEURON_BUFFER_ALLOCATOR_LIB environment variable is set
if [ -z "$NEURON_BUFFER_ALLOCATOR_LIB" ]; then
    echo "Error: NEURON_BUFFER_ALLOCATOR_LIB environment variable is not set." >&2
    exit 1
fi

# Create and enter the build directory
cd "$SOURCE_DIR"
rm -rf cmake-android-out && mkdir cmake-android-out && cd cmake-android-out

# Configure the project with CMake
# Note: Add any additional configuration options you need here
cmake -DBUCK2="$BUCK_PATH" \
      -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-30 \
      -DEXECUTORCH_BUILD_NEURON=ON \
      -DNEURON_BUFFER_ALLOCATOR_LIB="$NEURON_BUFFER_ALLOCATOR_LIB" \
      ..

# Build the project
cd ..
cmake --build cmake-android-out -j4

# Switch back to the original directory
cd - > /dev/null

# Print a success message
echo "Build successfully completed."
