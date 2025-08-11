#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
EXECUTORCH_ROOT=$(realpath "$(dirname "$0")/../..")
echo EXECUTORCH_ROOT=${EXECUTORCH_ROOT}

# Check if the ANDROID_NDK environment variable is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable is not set." >&2
    exit 1
fi

main() {
    # Enter the build directory
    cd "$EXECUTORCH_ROOT"

    # Set build directory
    local build_dir="cmake-android-out"

    # Check if the build directory exists
    if [ ! -d "$EXECUTORCH_ROOT/$build_dir" ]; then
        echo "Error: Build directory '$build_dir' does not exist."
        echo "Please build MTK backend before running this script."
        exit 1
    fi

    ## Build example
    local example_dir=examples/mediatek
    local example_build_dir="${build_dir}/${example_dir}"
    local cmake_prefix_path="${PWD}/${build_dir}/lib/cmake/ExecuTorch;${PWD}/${build_dir}/third-party/gflags;"
    rm -rf "${example_build_dir}"

    ## MTK original
    cmake -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
          -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI=arm64-v8a \
          -DANDROID_NATIVE_API_LEVEL=26 \
          -DANDROID_PLATFORM=android-26 \
          -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
          -B"${example_build_dir}" \
          $EXECUTORCH_ROOT/$example_dir

    cmake --build "${example_build_dir}" -j5

    # Switch back to the original directory
    cd - > /dev/null

    # Print a success message
    echo "Build successfully completed."
}

main "$@"
