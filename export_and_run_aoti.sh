#!/bin/bash

# Script to export and run AOTI with different modes
# Usage:
#   ./export_and_run_aoti.sh <model_arg> [mode]
#   ./export_and_run_aoti.sh <model_arg> --mode=<mode> [--debug] [--dump]
#
# Examples:
#   ./export_and_run_aoti.sh conv2d                         # Uses default mode (reinstall_all)
#   ./export_and_run_aoti.sh conv2d inference               # Uses inference mode
#   ./export_and_run_aoti.sh conv2d --mode=inference        # Alternative syntax
#   ./export_and_run_aoti.sh conv2d --mode=inference --dump # With AOTI intermediate output dumping
#   ./export_and_run_aoti.sh conv2d --mode=inference --debug --dump # With both debug and dump
#
# Available modes: reinstall_all (default), reinstall_aot, reinstall_runtime, inference, export_aoti_only
# Flags:
#   --debug: Enable debug mode with extensive logging
#   --dump:  Enable AOTI intermediate output dumping to aoti_intermediate_output.txt
# model_arg: argument to pass to export_aoti.py

set -e  # Exit on any error

# Parse command line arguments
MODE="reinstall_all"
MODEL_ARG="$1"
DEBUG_MODE=false
DUMP_MODE=false

# Parse arguments for mode and debug flag
for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE="${arg#*=}"
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --dump)
            DUMP_MODE=true
            shift
            ;;
        reinstall_all|reinstall_aot|reinstall_runtime|inference|export_aoti_only)
            # If it's the second argument and a valid mode, use it as mode
            if [[ "$arg" == "$2" ]]; then
                MODE="$arg"
            fi
            ;;
    esac
done

# Validate mode
case "$MODE" in
    reinstall_all|reinstall_aot|reinstall_runtime|inference|export_aoti_only)
        # Valid mode, continue
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Available modes: reinstall_all, reinstall_aot, reinstall_runtime, inference, export_aoti_only"
        echo ""
        echo "Usage examples:"
        echo "  ./export_and_run_aoti.sh conv2d                         # Uses default mode"
        echo "  ./export_and_run_aoti.sh conv2d inference               # Positional mode"
        echo "  ./export_and_run_aoti.sh conv2d --mode=inference        # GNU-style mode"
        echo "  ./export_and_run_aoti.sh conv2d export_aoti_only        # Export AOTI only (no runtime)"
        echo "  ./export_and_run_aoti.sh conv2d --mode=inference --debug # With debug options enabled"
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
    rm -f aoti_intermediate_output.txt

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
    local use_aoti_only=$1
    echo "Exporting AOTI model..."
    if [[ "$use_aoti_only" == "--aoti_only" ]]; then
        python export_aoti.py $MODEL_ARG --aoti_only
    else
        python export_aoti.py $MODEL_ARG
    fi
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

    if [[ "$DEBUG_MODE" == true ]]; then
        echo "Building with debug configuration..."
        cmake -DEXECUTORCH_BUILD_AOTI=ON \
              -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
              -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
              -DEXECUTORCH_LOG_LEVEL=Debug \
              -DCMAKE_BUILD_TYPE=Debug \
              ..
    else
        echo "Building with release configuration..."
        cmake -DEXECUTORCH_BUILD_AOTI=ON \
              -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
              -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
              -DEXECUTORCH_LOG_LEVEL=Info \
              -DCMAKE_BUILD_TYPE=Release \
              ..
    fi

    cd ..
    cmake --build cmake-out -j9
}

run_inference() {
    echo "Running executor_runner with debug logging enabled..."
    ./cmake-out/executor_runner --model_path aoti_model.pte
}

compare_outputs() {
    echo "Comparing runtime outputs with label outputs..."
    python compare_outputs.py
}

# Set up environment variables based on debug and dump flags
if [[ "$DEBUG_MODE" == true ]]; then
    echo "Setting debug environment variables..."
    export AOT_INDUCTOR_DEBUG_COMPILE="1"
    export AOTINDUCTOR_REPRO_LEVEL=3

    # Set intermediate value printer based on dump flag
    if [[ "$DUMP_MODE" == true ]]; then
        export AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER="2"
        export INDUCTOR_PROVENANCE=1
        export TORCH_TRACE="/home/gasoonjia/executorch/aoti_debug_data"
        echo "AOTI intermediate output dumping enabled (AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2)"
        echo "Eager-AOTI relationship extration enabled (INDUCTOR_PROVENANCE=1), output to $TORCH_TRACE"
    else
        export AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER="3"
    fi

    echo "Debug variables set:"
    echo "  AOT_INDUCTOR_DEBUG_COMPILE=$AOT_INDUCTOR_DEBUG_COMPILE"
    echo "  AOTINDUCTOR_REPRO_LEVEL=$AOTINDUCTOR_REPRO_LEVEL"
    echo "  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=$AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER"
elif [[ "$DUMP_MODE" == true ]]; then
    # Only dump mode enabled (without debug)
    echo "Setting AOTI intermediate output dumping..."
    export AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER="2"
    export INDUCTOR_PROVENANCE=1
    export TORCH_TRACE="/home/gasoonjia/executorch/aoti_debug_data"
    echo "AOTI intermediate output dumping enabled (AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2)"
    echo "  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=$AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER"
    echo "Eager-AOTI relationship extration enabled (INDUCTOR_PROVENANCE=1), output to $TORCH_TRACE"
else
    # Ensure debug variables are unset for non-debug/non-dump modes
    unset AOT_INDUCTOR_DEBUG_COMPILE
    unset AOTINDUCTOR_REPRO_LEVEL
    unset AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER
    unset INDUCTOR_PROVENANCE
    unset TORCH_TRACE
fi

# Execute based on mode
case "$MODE" in
    "reinstall_all")
        echo "Mode: $MODE - Full reinstall and run"
        if [[ "$DEBUG_MODE" == true ]]; then
            echo "Debug options enabled with AOT Inductor debug settings"
        fi
        install_executorch
        export_aoti_model
        clean_install_executorch
        build_runtime
        run_inference
        compare_outputs
        ;;
    "reinstall_aot")
        echo "Mode: reinstall_aot - Reinstall AOT components and run e2e"
        if [[ "$DEBUG_MODE" == true ]]; then
            echo "Debug options enabled with AOT Inductor debug settings"
        fi
        install_executorch
        export_aoti_model
        run_inference
        compare_outputs
        ;;
    "reinstall_runtime")
        echo "Mode: reinstall_runtime - Rebuild runtime and run e2e"
        if [[ "$DEBUG_MODE" == true ]]; then
            echo "Debug options enabled with AOT Inductor debug settings"
        fi
        export_aoti_model
        build_runtime
        run_inference
        compare_outputs
        ;;
    "inference")
        echo "Mode: inference - Export model and run inference only"
        if [[ "$DEBUG_MODE" == true ]]; then
            echo "Debug options enabled with AOT Inductor debug settings"
        fi
        export_aoti_model
        run_inference
        compare_outputs
        ;;
    "export_aoti_only")
        echo "Mode: export_aoti_only - Export model using pure AOTI only (no runtime or installation)"
        if [[ "$DEBUG_MODE" == true ]]; then
            echo "Debug options enabled with AOT Inductor debug settings"
        fi
        export_aoti_model "--aoti_only"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Available modes: reinstall_all, reinstall_aot, reinstall_runtime, inference, export_aoti_only"
        exit 1
        ;;
esac

echo "Script completed successfully!"
