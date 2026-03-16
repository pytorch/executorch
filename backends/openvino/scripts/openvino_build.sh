#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where CMakeLists.txt is located
EXECUTORCH_ROOT=$(realpath "$(dirname "$0")/../../..")
echo EXECUTORCH_ROOT=${EXECUTORCH_ROOT}

# Enter the Executorch root directory
cd "$EXECUTORCH_ROOT"

install_requirements() {
    echo "Installing Requirements For OpenVINO Backend"
    pip install -r backends/openvino/requirements.txt
}

build_cpp_runtime() {
    echo "Building C++ Runtime Libraries"

    local llm_enabled=${1:-0}

    # Set build directory
    local build_dir="cmake-out"

    rm -rf "${build_dir}"

    CMAKE_ARGS=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DEXECUTORCH_BUILD_OPENVINO=ON"
        "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON"
        "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON"
        "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON"
        "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON"
        "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON"
        "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON"
        "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON"
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON"
    )

    if [[ "$llm_enabled" -eq 1 ]]; then
        CMAKE_ARGS+=("-DEXECUTORCH_BUILD_EXTENSION_LLM=ON -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON")
    fi

    # Configure the project with CMake
    # Note: Add any additional configuration options you need here
    cmake -DCMAKE_INSTALL_PREFIX="${build_dir}" \
          ${CMAKE_ARGS[@]} \
          -B"${build_dir}"

    # Build the project
    cmake --build ${build_dir} --target install --config Release -j$(nproc)
}

build_python_enabled() {
    echo "Building Python Package with Pybinding"

    # Enter the Executorch root directory

    # Set parameters to configure the project with CMake
    # Note: Add any additional configuration options you need here
    export CMAKE_ARGS="-DEXECUTORCH_BUILD_OPENVINO=ON \
                       -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON"
    export CMAKE_BUILD_ARGS="--target openvino_backend"

    # Build the package
    ./install_executorch.sh --use-pt-pinned-commit
}

main() {
    build_type=${1:-"--build_all"}

    # If the first argument is --build_all (default), build python package, C++ runtime
    if [[ -z "$build_type" || "$build_type" == "--build_all" ]]; then
        ./install_executorch.sh --clean
        install_requirements
        build_python_enabled
        build_cpp_runtime

    # If the first argument is --cpp_runtime, build libraries for C++ runtime
    elif [[ "$build_type" == "--cpp_runtime" ]]; then
        build_cpp_runtime

    # If the first argument is --cpp_runtime_llm, build C++ runtime with llm extension
    # Note: c++ runtime with openvino backend should be built before building llama runner
    elif [[ "$build_type" == "--cpp_runtime_llm" ]]; then
        build_cpp_runtime 1

    # If the first argument is --enable_python, build python package with python bindings
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
