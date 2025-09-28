#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Prerequisite steps: (run the following commands before running this script)
# 1. Setup your environment for Arm FVP
#   a. Setup Conda environment / venv
#   b. ./install_executorch.sh --clean ; ./install_executorch.sh --editable;
#   c. examples/arm/setup.sh --i-agree-to-the-contained-eula;
#   d. source examples/arm/ethos-u-scratch/setup_path.sh
# 2. bash examples/selective_build/test_selective_build.sh cmake

set -u

# Valid targets for MCU model validation
VALID_TARGETS=(
    "cortex-m55"
    "cortex-m85"
)

# Default models for MCU validation with portable kernels
DEFAULT_MODELS=(mv2 mv3 lstm qadd qlinear)
# Available models (on FVP)
AVAILABLE_MODELS=(mv2 mv3 lstm qadd qlinear)
# Add the following models if you want to enable them later (atm they are not working on FVP)
# edsr w2l ic3 ic4 resnet18 resnet50

# Variables
TARGET=""
MODELS=()
PASSED_MODELS=()
FAILED_MODELS=()

# Function to validate target
validate_target() {
    local target=$1
    for valid_target in "${VALID_TARGETS[@]}"; do
        if [[ "$target" == "$valid_target" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to validate models
validate_models() {
    local invalid_models=()
    for model in "${MODELS[@]}"; do
        if [[ ! " ${AVAILABLE_MODELS[*]} " =~ " $model " ]]; then
            invalid_models+=("$model")
        fi
    done

    if [[ ${#invalid_models[@]} -gt 0 ]]; then
        echo "❌ Error: Invalid model(s): ${invalid_models[*]}"
        echo "Available models: ${AVAILABLE_MODELS[*]}"
        return 1
    fi
    return 0
}

cpu_to_ethos_target() {
  local cpu=$1
  case $cpu in
    cortex-m55)
      echo "ethos-u55-128"
      ;;
    cortex-m85)
      echo "ethos-u85-128"
      ;;
    *)
      echo "Unknown CPU: $cpu" >&2
      return 1
      ;;
  esac
}

# Function to show usage
show_usage() {
    echo "Usage: $0 --target=<target> [--models=<model1,model2,...>]"
    echo ""
    echo "MCU Model Validation without delegation"
    echo ""
    echo "Required arguments:"
    echo "  --target=<target>         Target platform for validation"
    echo ""
    echo "Optional arguments:"
    echo "  --models=<models>         Comma-separated list of models to test"
    echo "                           (overrides default model list)"
    echo ""
    echo "Valid targets:"
    printf '  %s\n' "${VALID_TARGETS[@]}"
    echo ""
    echo "Available models:"
    printf '  %s\n' "${AVAILABLE_MODELS[@]}"
    echo ""
    echo "Examples:"
    echo "  $0 --target=ethos-u85-128"
    echo "  $0 --target=ethos-u55-128 --models=mv2,mv3,resnet18"
    echo ""
    echo "Default behavior:"
    echo "  - Uses all available models: ${DEFAULT_MODELS[*]}"
    echo "  - Runs with portable kernels (no delegation)"
}

# Function to display summary
show_summary() {
    local total_models=${#MODELS[@]}

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "🏁 MCU MODEL VALIDATION SUMMARY - TARGET: $TARGET"
    echo "════════════════════════════════════════════════════════════════"
    echo ""

    # Show individual results
    for model in "${MODELS[@]}"; do
        if [[ " ${PASSED_MODELS[*]} " =~ " $model " ]]; then
            printf "%-12s : ✅ Passed\n" "$model"
        elif [[ " ${FAILED_MODELS[*]} " =~ " $model " ]]; then
            printf "%-12s : ❌ Failed\n" "$model"
        else
            printf "%-12s : ⏭️  Skipped\n" "$model"
        fi
    done

    echo ""
    echo "────────────────────────────────────────────────────────────────"

    # Show statistics
    local passed_count=${#PASSED_MODELS[@]}
    local failed_count=${#FAILED_MODELS[@]}
    local success_rate=$((passed_count * 100 / total_models))

    echo "📊 STATISTICS:"
    echo "   Total Models    : $total_models"
    echo "   ✅ Passed       : $passed_count"
    echo "   ❌ Failed       : $failed_count"
    echo "   📈 Success Rate : $success_rate%"
    echo ""

    # Show model selection info
    if [[ ${#MODELS[@]} -eq ${#DEFAULT_MODELS[@]} ]] && [[ "${MODELS[*]}" == "${DEFAULT_MODELS[*]}" ]]; then
        echo "📋 Model Selection: Default (all available models)"
    else
        echo "📋 Model Selection: Custom (${MODELS[*]})"
    fi
    echo ""

    # Overall result
    if [[ $failed_count -eq 0 ]]; then
        echo "🎉 OVERALL RESULT: ALL TESTS PASSED!"
        echo "🔧 Mode: Portable Kernels (No Delegation)"
    else
        echo "⚠️  OVERALL RESULT: $failed_count/$total_models TESTS FAILED"
        echo "🔧 Mode: Portable Kernels (No Delegation)"
        echo ""
        echo "🔍 Failed models: ${FAILED_MODELS[*]}"
    fi

    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target=*)
            TARGET="${1#*=}"
            shift
            ;;
        --models=*)
            IFS=',' read -ra MODELS <<< "${1#*=}"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Error: Unknown argument '$1'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
done

# Check if target is provided
if [[ -z "$TARGET" ]]; then
    echo "❌ Error: --target argument is required"
    echo ""
    show_usage
    exit 1
fi

# Validate target
if ! validate_target "$TARGET"; then
    echo "❌ Error: Invalid target '$TARGET'"
    echo ""
    show_usage
    exit 1
fi

# Use default models if none specified
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Validate models
if ! validate_models; then
    exit 1
fi

# Remove duplicates from models array
IFS=" " read -r -a MODELS <<< "$(printf '%s\n' "${MODELS[@]}" | sort -u | tr '\n' ' ')"

echo "🎯 MCU Model Validation - Target: $TARGET"
echo "📋 Processing models: ${MODELS[*]}"
echo "🔧 Mode: Portable Kernels (No Delegation)"
echo ""

echo "🔨 Building ExecuteTorch libraries (one-time setup)..."
if ! backends/arm/scripts/build_executorch.sh; then
    echo "❌ Failed to build ExecuteTorch libraries"
    exit 1
fi
echo "✅ ExecuteTorch libraries built successfully"
echo ""

ETHOS_TARGET=$(cpu_to_ethos_target "$TARGET")
if [[ $? -ne 0 ]]; then
    echo "Invalid CPU target: $TARGET"
    exit 1
fi
echo "Using ETHOS target: $ETHOS_TARGET"

# Process each model
for model in "${MODELS[@]}"; do
    echo "=== 🚀 Processing $model for $TARGET ==="

    # Track if this model succeeds
    MODEL_SUCCESS=true

    # Step 1: Create directory
    echo "📁 Creating directory arm_test/$model"
    mkdir -p "arm_test/$model"

    # Step 2: AOT compilation (quantized, no delegation = portable kernels)
    echo "⚙️  AOT compilation for $model"
    if ! python3 -m examples.arm.aot_arm_compiler \
        -m "$model" \
        --target="$ETHOS_TARGET" \
        --quantize \
        --enable_qdq_fusion_pass \
        --output="arm_test/$model"; then
        echo "❌ AOT compilation failed for $model"
        MODEL_SUCCESS=false
    fi

    # Step 3: Build executor runner (only if AOT succeeded)
    if [[ "$MODEL_SUCCESS" == true ]]; then
        echo "🔨 Building executor runner for $model"
        if ! backends/arm/scripts/build_executor_runner.sh \
            --pte="arm_test/$model/${model}_arm_${ETHOS_TARGET}.pte" \
            --target="$ETHOS_TARGET" \
            --output="arm_test/$model"; then
            echo "❌ Executor runner build failed for $model"
            MODEL_SUCCESS=false
        fi
    fi

    # Step 4: Run on FVP (only if build succeeded)
    if [[ "$MODEL_SUCCESS" == true ]]; then
        echo "🏃 Running $model on FVP with portable kernels"
        if ! backends/arm/scripts/run_fvp.sh \
            --elf="arm_test/$model/arm_executor_runner" \
            --target="$ETHOS_TARGET"; then
            echo "❌ FVP execution failed for $model"
            MODEL_SUCCESS=false
        fi
    fi

    # Record result
    if [[ "$MODEL_SUCCESS" == true ]]; then
        echo "✅ $model completed successfully"
        PASSED_MODELS+=("$model")
    else
        echo "❌ $model failed"
        FAILED_MODELS+=("$model")
    fi

    echo ""
done

# Show comprehensive summary
show_summary

# Exit with appropriate code for CI
if [[ ${#FAILED_MODELS[@]} -eq 0 ]]; then
    exit 0  # Success
else
    exit 1  # Failure
fi
