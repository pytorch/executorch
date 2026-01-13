#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
EXECUTORCH_ROOT=$(realpath "$(dirname "$0")/../../..")
echo EXECUTORCH_ROOT=${EXECUTORCH_ROOT}

install_requirements() {
    echo "Installing Requirements For OpenVINO Backend"
    cd "$EXECUTORCH_ROOT"
    pip install -r backends/openvino/requirements.txt
}

build_cpp_runtime() {
    echo "Building C++ Runtime Libraries"

    # Set build directory
    local build_dir="cmake-out"

    # Enter the Executorch root directory
    cd "$EXECUTORCH_ROOT"
    rm -rf "${build_dir}"

    # Configure the project with CMake
    # Note: Add any additional configuration options you need here
    cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_OPENVINO=ON \
          -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
          -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
          -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
          -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
          -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
          -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
          -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
          -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
          -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
          -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
          -B"${build_dir}"


    # Build the project
    cmake --build ${build_dir} --target install --config Release -j$(nproc)
}

build_llama_runner() {
    echo "Building Export Llama Runner"

    # Set build directory
    local build_dir="cmake-out"

    # Enter the Executorch root directory
    cd "$EXECUTORCH_ROOT"

    # Configure the project with CMake
    # Note: Add any additional configuration options you need here
    cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -B"${build_dir}"/examples/models/llama \
        examples/models/llama
    # Build the export llama runner
    cmake --build cmake-out/examples/models/llama -j$(nproc) --config Release
}

build_python_enabled() {
    echo "Building Python Package with Pybinding"

    # Enter the Executorch root directory
    cd "$EXECUTORCH_ROOT"
    ./install_executorch.sh --clean

    # Set parameters to configure the project with CMake
    # Note: Add any additional configuration options you need here
    export CMAKE_ARGS="-DEXECUTORCH_BUILD_OPENVINO=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON"
    export CMAKE_BUILD_ARGS="--target openvino_backend"

    # Build the package
    ./install_executorch.sh --minimal

    # Install torchao
    # Note: --no-build-isolation is required because torchao's setup.py imports torch
    # See comment in torchao's pyproject.toml for more details
    pip install third-party/ao --no-build-isolation
}

main() {
    build_type=${1:-"--build_all"}

    # If the first arguments is --build_all (default), build python package, C++ runtime, and llama runner binary
    if [[ -z "$build_type" || "$build_type" == "--build_all" ]]; then
        install_requirements
        build_python_enabled
        build_cpp_runtime
        build_llama_runner

    # If the first arguments is --cpp_runtime, build libraries for C++ runtime
    elif [[ "$build_type" == "--cpp_runtime" ]]; then
        build_cpp_runtime

    # If the first arguments is --llama_runner, build export llama runner binary
    # Note: c++ runtime with openvino backend should be built before building export llama runner
    elif [[ "$build_type" == "--llama_runner" ]]; then
        build_llama_runner

    # If the first arguments is --enable_python, build python package with python bindings
    elif [[ "$build_type" == "--enable_python" ]]; then
        install_requirements
        build_python_enabled

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
