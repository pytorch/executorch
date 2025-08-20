#!/bin/bash

# Script to export and run AOTI with different modes
# Usage:
#   ./export_and_run_aoti.sh <model_arg> [mode]
#   ./export_and_run_aoti.sh <model_arg> --mode=<mode>
#
# Examples:
#   ./export_and_run_aoti.sh conv2d                    # Uses default mode (reinstall_all)
#   ./export_and_run_aoti.sh conv2d inference          # Uses inference mode
#   ./export_and_run_aoti.sh conv2d --mode=inference   # Alternative syntax
#
# Available modes: reinstall_all (default), reinstall_aot, reinstall_runtime, inference
# model_arg: argument to pass to export_aoti.py

set -e  # Exit on any error

# Parse command line arguments
MODE="reinstall_all"
MODEL_ARG="$1"

# Parse arguments for mode
for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE="${arg#*=}"
            shift
            ;;
        reinstall_all|reinstall_aot|reinstall_runtime|inference)
            # If it's the second argument and a valid mode, use it as mode
            if [[ "$arg" == "$2" ]]; then
                MODE="$arg"
            fi
            ;;
    esac
done

# Validate mode
case "$MODE" in
    reinstall_all|reinstall_aot|reinstall_runtime|inference)
        # Valid mode, continue
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Available modes: reinstall_all, reinstall_aot, reinstall_runtime, inference"
        echo ""
        echo "Usage examples:"
        echo "  ./export_and_run_aoti.sh conv2d                    # Uses default mode"
        echo "  ./export_and_run_aoti.sh conv2d inference          # Positional mode"
        echo "  ./export_and_run_aoti.sh conv2d --mode=inference   # GNU-style mode"
        exit 1
        ;;
esac

echo "Running in mode: $MODE"
if [[ -n "$MODEL_ARG" ]]; then
    echo "Model argument: $MODEL_ARG"
fi

# Cleanup function to remove temporary files and directories
cleanup_temp_files() {
    echo "Cleaning up temporary files and directories..."

    # Remove temporary directories
    for file in *wrapper.cpp; do
        if [[ -f "$file" ]]; then
            basename="${file%wrapper.cpp}"
            if [[ -d "$basename" ]]; then
                echo "Removing directory: $basename"
                rm -rf "$basename"
            fi
        fi
    done

    # Remove temporary files with specific extensions
    rm -f *.cubin
    rm -f *.pte
    rm -f *.so
    rm -f *kernel_metadata.json
    rm -f *kernel.cpp
    rm -f *wrapper_metadata.json
    rm -f *wrapper.cpp

    echo "Cleanup completed."
}

# Run cleanup at the start
cleanup_temp_files

# Function definitions for each step
install_executorch() {
    echo "Installing executorch..."
    ./install_executorch.sh
}

export_aoti_model() {
    echo "Exporting AOTI model..."
    python export_aoti.py $MODEL_ARG
}

clean_install_executorch() {
    echo "Clean installing executorch..."
    ./install_executorch.sh --clean
}

build_runtime() {
    echo "Building runtime..."
    # Clean the build directory to ensure debug flags take effect
    rm -rf cmake-out
    mkdir -p cmake-out
    cd cmake-out
    cmake -DEXECUTORCH_BUILD_AOTI=ON \
          -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
          -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
          -DEXECUTORCH_LOG_LEVEL=Debug \
          -DCMAKE_BUILD_TYPE=Debug \
          ..
    cd ..
    cmake --build cmake-out -j9
}

run_inference() {
    echo "Running executor_runner with debug logging enabled..."
    ./cmake-out/executor_runner --model_path aoti_model.pte
}

# Execute based on mode
case "$MODE" in
    "reinstall_all")
        echo "Mode: reinstall_all - Full reinstall and run"
        install_executorch
        export_aoti_model
        clean_install_executorch
        build_runtime
        run_inference
        ;;
    "reinstall_aot")
        echo "Mode: reinstall_aot - Reinstall AOT components and run e2e"
        install_executorch
        export_aoti_model
        run_inference
        ;;
    "reinstall_runtime")
        echo "Mode: reinstall_runtime - Rebuild runtime and run e2e"
        export_aoti_model
        build_runtime
        run_inference
        ;;
    "inference")
        echo "Mode: inference - Export model and run inference only"
        export_aoti_model
        run_inference
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Available modes: reinstall_all, reinstall_aot, reinstall_runtime, inference"
        exit 1
        ;;
esac

echo "Script completed successfully!"
