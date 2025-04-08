#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
EXECUTORCH_ROOT=$(realpath "$(dirname "$0")/../../..")
echo EXECUTORCH_ROOT=${EXECUTORCH_ROOT}

main() {
    build_type=${1:-"--cpp_runtime"}

    # If the first arguments is --cpp_runtime (default), build libraries for C++ runtime
    if [[ -z "$build_type" || "$build_type" == "--cpp_runtime" ]]; then
        echo "Building C++ Runtime Libraries"

        # Set build directory
        local build_dir="cmake-out"

        # Create and enter the build directory
        cd "$EXECUTORCH_ROOT"
        rm -rf "${build_dir}"

        # Configure the project with CMake
        # Note: Add any additional configuration options you need here
        cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
              -DCMAKE_BUILD_TYPE=Release \
              -DEXECUTORCH_BUILD_OPENVINO=ON \
              -DEXECUTORCH_BUILD_XNNPACK=ON \
              -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
              -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
              -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
              -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
              -DEXECUTORCH_BUILD_OPENVINO_EXECUTOR_RUNNER=ON \
              -DPYTHON_EXECUTABLE=python \
              -B"${build_dir}"
              #-DEXECUTORCH_LOG_LEVEL=Debug \


        # Build the project
        cmake --build ${build_dir} --target install --config Release -j$(nproc)

    # If the first arguments is --enable_python, build python package with python bindings
    elif [[ "$build_type" == "--enable_python" ]]; then
        echo "Building Python Package with Pybinding"

        # Create and enter the build directory
        cd "$EXECUTORCH_ROOT"
        ./install_executorch.sh --clean

        # Set parameters to configure the project with CMake
        # Note: Add any additional configuration options you need here
        export CMAKE_ARGS="-DEXECUTORCH_BUILD_OPENVINO=ON \
                           -DEXECUTORCH_BUILD_XNNPACK=ON \
                           -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
                           -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
                           -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
                           -DEXECUTORCH_ENABLE_LOGGING=ON \
                           -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
                           -DEXECUTORCH_BUILD_PYBIND=ON"
        export CMAKE_BUILD_ARGS="--target openvino_backend xnnpack_backend"

        # Build the package
        pip install . --no-build-isolation

    else
        echo "Error: Argument is not valid: $build_type"
        exit 1  # Exit the script with an error code
    fi

    # Switch back to the original directory
    cd - > /dev/null

    # Print a success message
    echo "Build successfully completed."

}

main "$@"
