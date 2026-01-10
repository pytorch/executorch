#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build and run the static LLM C++ runner for CoreML
#
# Usage:
#   ./build_and_run.sh [--rebuild] [--run-only] [--help]
#
# Arguments:
#   --rebuild    Force a clean rebuild
#   --run-only   Skip build, just run the executable
#   --help       Show this help message
#
# Environment variables (override defaults):
#   MODEL_PATH      Path to the .pte model file
#   PARAMS_PATH     Path to params.json
#   TOKENIZER_PATH  Path to tokenizer.model
#   PROMPT          Input prompt
#   MAX_NEW_TOKENS  Maximum tokens to generate
#   INPUT_LEN       Input sequence length
#   CACHE_LEN       KV cache length

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out"

# Default values (can be overridden via environment variables)
MODEL_PATH="${MODEL_PATH:-$HOME/Desktop/static_llama1b_coreml_model.pte}"
PARAMS_PATH="${PARAMS_PATH:-$HOME/models/llama1b/params.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/models/llama1b/tokenizer.model}"
PROMPT="${PROMPT:-Once upon a time,}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
INPUT_LEN="${INPUT_LEN:-32}"
CACHE_LEN="${CACHE_LEN:-992}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# Lookahead decoding options
LOOKAHEAD="${LOOKAHEAD:-false}"
NGRAM_SIZE="${NGRAM_SIZE:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-5}"
N_VERIFICATIONS="${N_VERIFICATIONS:-3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [--rebuild] [--run-only] [--help]"
    echo ""
    echo "Options:"
    echo "  --rebuild    Force a clean rebuild"
    echo "  --run-only   Skip build, just run the executable"
    echo "  --help       Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_PATH      Path to the .pte model file (default: \$HOME/Desktop/static_llama1b_coreml_model.pte)"
    echo "  PARAMS_PATH     Path to params.json (default: \$HOME/models/llama1b/params.json)"
    echo "  TOKENIZER_PATH  Path to tokenizer.model (default: \$HOME/models/llama1b/tokenizer.model)"
    echo "  PROMPT          Input prompt (default: 'Once upon a time,')"
    echo "  MAX_NEW_TOKENS  Maximum tokens to generate (default: 100)"
    echo "  INPUT_LEN       Input sequence length (default: 32)"
    echo "  CACHE_LEN       KV cache length (default: 992)"
    echo "  TEMPERATURE     Sampling temperature (default: 0.0)"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
REBUILD=false
RUN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required files exist
validate_files() {
    local missing=false

    if [[ ! -f "${MODEL_PATH}" ]]; then
        log_error "Model file not found: ${MODEL_PATH}"
        missing=true
    fi

    if [[ ! -f "${PARAMS_PATH}" ]]; then
        log_error "Params file not found: ${PARAMS_PATH}"
        missing=true
    fi

    if [[ ! -f "${TOKENIZER_PATH}" ]]; then
        log_error "Tokenizer file not found: ${TOKENIZER_PATH}"
        missing=true
    fi

    if [[ "$missing" == "true" ]]; then
        echo ""
        log_info "You can set paths via environment variables:"
        echo "  export MODEL_PATH=/path/to/model.pte"
        echo "  export PARAMS_PATH=/path/to/params.json"
        echo "  export TOKENIZER_PATH=/path/to/tokenizer.model"
        exit 1
    fi
}

build_project() {
    log_info "ExecutorTorch root: ${EXECUTORCH_ROOT}"
    log_info "Build directory: ${BUILD_DIR}"

    # Clean build if requested
    if [[ "$REBUILD" == "true" ]] && [[ -d "${BUILD_DIR}" ]]; then
        log_info "Cleaning build directory..."
        rm -rf "${BUILD_DIR}"
    fi

    cd "${EXECUTORCH_ROOT}"

    # Configure CMake using macos preset (includes all necessary LLM extensions)
    log_info "Configuring CMake with macos preset..."
    cmake -S "${EXECUTORCH_ROOT}" \
          -B "${BUILD_DIR}" \
          -DCMAKE_BUILD_TYPE=Release \
          --preset macos

    # Build the target
    log_info "Building run_static_llm_coreml..."
    cmake --build "${BUILD_DIR}" \
          -j$(sysctl -n hw.ncpu) \
          --config Release \
          --target run_static_llm_coreml

    log_info "Build complete!"
}

run_model() {
    local executable="${BUILD_DIR}/examples/apple/coreml/llama/runner/Release/run_static_llm_coreml"

    # Also check non-Release location
    if [[ ! -f "${executable}" ]]; then
        executable="${BUILD_DIR}/examples/apple/coreml/llama/runner/run_static_llm_coreml"
    fi

    # Check Debug location
    if [[ ! -f "${executable}" ]]; then
        executable="${BUILD_DIR}/examples/apple/coreml/llama/runner/Debug/run_static_llm_coreml"
    fi

    if [[ ! -f "${executable}" ]]; then
        log_error "Executable not found: ${executable}"
        log_info "Run without --run-only to build first"
        exit 1
    fi

    log_info "Running model..."
    echo ""
    echo "Configuration:"
    echo "  Model:          ${MODEL_PATH}"
    echo "  Params:         ${PARAMS_PATH}"
    echo "  Tokenizer:      ${TOKENIZER_PATH}"
    echo "  Prompt:         ${PROMPT}"
    echo "  Max tokens:     ${MAX_NEW_TOKENS}"
    echo "  Input length:   ${INPUT_LEN}"
    echo "  Cache length:   ${CACHE_LEN}"
    echo "  Temperature:    ${TEMPERATURE}"
    if [[ "${LOOKAHEAD}" == "true" ]]; then
        echo "  Lookahead:      enabled"
        echo "    ngram_size:   ${NGRAM_SIZE}"
        echo "    window_size:  ${WINDOW_SIZE}"
        echo "    n_verifications: ${N_VERIFICATIONS}"
    fi
    echo ""
    echo "=========================================="

    # Build command with optional lookahead flags
    local cmd=("${executable}"
        --model "${MODEL_PATH}"
        --params "${PARAMS_PATH}"
        --tokenizer "${TOKENIZER_PATH}"
        --prompt "${PROMPT}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}")

    if [[ "${LOOKAHEAD}" == "true" ]]; then
        cmd+=(--lookahead
              --ngram_size "${NGRAM_SIZE}"
              --window_size "${WINDOW_SIZE}"
              --n_verifications "${N_VERIFICATIONS}")
    fi

    "${cmd[@]}"

    echo "=========================================="
    log_info "Done!"
}

# Main execution
main() {
    echo ""
    log_info "Static LLM CoreML Runner - Build & Test Script"
    echo ""

    # Validate files before running
    validate_files

    if [[ "$RUN_ONLY" == "false" ]]; then
        build_project
    fi

    run_model
}

main
