#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
EXECUTORCH_ROOT=$(realpath "$(dirname "$0")/../..")
echo EXECUTORCH_ROOT=${EXECUTORCH_ROOT}

main() {
    # Set build directory
    local build_dir="cmake-openvino-out"

    # Create and enter the build directory
    cd "$EXECUTORCH_ROOT"
    rm -rf "${build_dir}"

    # Configure the project with CMake
    # Note: Add any additional configuration options you need here
    cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_OPENVINO=ON \
          -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
          -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
          -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
          -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
          -B"${build_dir}"


    # Build the project
    cmake --build cmake-openvino-out --target install --config Release -j$(nproc)

    ## Build example
    local example_dir=examples/openvino
    local example_build_dir="${build_dir}/${example_dir}"
    local cmake_prefix_path="${PWD}/${build_dir}/lib/cmake/ExecuTorch;${PWD}/${build_dir}/third-party/gflags;"
    rm -rf "${example_build_dir}"

    ## OpenVINO original
    cmake -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
          -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
          -B"${example_build_dir}" \
          $EXECUTORCH_ROOT/$example_dir

    cmake --build "${example_build_dir}" -j$(nproc)

    # Switch back to the original directory
    cd - > /dev/null

    # Print a success message
    echo "Build successfully completed."
}

main "$@"
