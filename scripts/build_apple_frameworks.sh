#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

MODES=()
PRESETS=("ios" "ios-simulator" "macos")
# To support backwards compatibility, we want to retain the same output directory.
PRESETS_RELATIVE_OUT_DIR=("ios" "simulator" "macos")

SOURCE_ROOT_DIR=$(git rev-parse --show-toplevel)
OUTPUT_DIR="${SOURCE_ROOT_DIR}/cmake-out"

BUCK2=$(python3 "$SOURCE_ROOT_DIR/tools/cmake/resolve_buck.py" --cache_dir="$SOURCE_ROOT_DIR/buck2-bin")
if [[ "$BUCK2" == "buck2" ]]; then
  BUCK2=$(command -v buck2)
fi

FRAMEWORK_EXECUTORCH_NAME="executorch"
FRAMEWORK_EXECUTORCH_MODULE_NAME="ExecuTorch"
FRAMEWORK_EXECUTORCH_HEADERS_DIR="${FRAMEWORK_EXECUTORCH_NAME}_include"
FRAMEWORK_EXECUTORCH_HEADERS_PATH="${OUTPUT_DIR}/${FRAMEWORK_EXECUTORCH_HEADERS_DIR}"
FRAMEWORK_EXECUTORCH="${FRAMEWORK_EXECUTORCH_NAME}:\
libexecutorch.a,\
libexecutorch_core.a,\
libextension_apple.a,\
libextension_data_loader.a,\
libextension_flat_tensor.a,\
libextension_module.a,\
libextension_tensor.a,\
:${FRAMEWORK_EXECUTORCH_HEADERS_DIR}:${FRAMEWORK_EXECUTORCH_MODULE_NAME}"

FRAMEWORK_EXECUTORCH_LLM_NAME="executorch_llm"
FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME="ExecuTorchLLM"
FRAMEWORK_EXECUTORCH_LLM_HEADERS_DIR="${FRAMEWORK_EXECUTORCH_LLM_NAME}_include"
FRAMEWORK_EXECUTORCH_LLM_HEADERS_PATH="${OUTPUT_DIR}/${FRAMEWORK_EXECUTORCH_LLM_HEADERS_DIR}"
FRAMEWORK_EXECUTORCH_LLM="${FRAMEWORK_EXECUTORCH_LLM_NAME}:\
libabsl_base.a,\
libabsl_city.a,\
libabsl_decode_rust_punycode.a,\
libabsl_demangle_internal.a,\
libabsl_demangle_rust.a,\
libabsl_examine_stack.a,\
libabsl_graphcycles_internal.a,\
libabsl_hash.a,\
libabsl_int128.a,\
libabsl_kernel_timeout_internal.a,\
libabsl_leak_check.a,\
libabsl_log_globals.a,\
libabsl_log_internal_check_op.a,\
libabsl_log_internal_format.a,\
libabsl_log_internal_globals.a,\
libabsl_log_internal_log_sink_set.a,\
libabsl_log_internal_message.a,\
libabsl_log_internal_nullguard.a,\
libabsl_log_internal_proto.a,\
libabsl_log_severity.a,\
libabsl_log_sink.a,\
libabsl_low_level_hash.a,\
libabsl_malloc_internal.a,\
libabsl_raw_hash_set.a,\
libabsl_raw_logging_internal.a,\
libabsl_spinlock_wait.a,\
libabsl_stacktrace.a,\
libabsl_str_format_internal.a,\
libabsl_strerror.a,\
libabsl_strings.a,\
libabsl_strings_internal.a,\
libabsl_symbolize.a,\
libabsl_synchronization.a,\
libabsl_throw_delegate.a,\
libabsl_time.a,\
libabsl_time_zone.a,\
libabsl_tracing_internal.a,\
libabsl_utf8_for_code_point.a,\
libextension_llm_apple.a,\
libextension_llm_runner.a,\
libpcre2-8.a,\
libre2.a,\
libregex_lookahead.a,\
libsentencepiece.a,\
libtokenizers.a,\
:${FRAMEWORK_EXECUTORCH_LLM_HEADERS_DIR}"

FRAMEWORK_THREADPOOL="threadpool:\
libcpuinfo.a,\
libextension_threadpool.a,\
libpthreadpool.a,\
:"

FRAMEWORK_BACKEND_COREML="backend_coreml:\
libcoreml_util.a,\
libcoreml_inmemoryfs.a,\
libcoremldelegate.a,\
:"

FRAMEWORK_BACKEND_MPS="backend_mps:\
libmpsdelegate.a,\
:"

FRAMEWORK_BACKEND_XNNPACK="backend_xnnpack:\
libXNNPACK.a,\
libkleidiai.a,\
libxnnpack_backend.a,\
libxnnpack-microkernels-prod.a,\
:"

FRAMEWORK_KERNELS_LLM="kernels_llm:\
libcustom_ops.a,\
:"

FRAMEWORK_KERNELS_OPTIMIZED="kernels_optimized:\
libcpublas.a,\
liboptimized_kernels.a,\
liboptimized_native_cpu_ops_lib.a,\
libportable_kernels.a,\
:"

FRAMEWORK_KERNELS_QUANTIZED="kernels_quantized:\
libquantized_kernels.a,\
libquantized_ops_lib.a,\
:"

FRAMEWORK_KERNELS_TORCHAO="kernels_torchao:\
libtorchao_ops_executorch.a,\
libtorchao_kernels_aarch64.a,\
:"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Build frameworks for Apple platforms."
  echo
  echo "Options:"
  echo "  --Debug              Build Debug version."
  echo "  --Release            Build Release version."
  echo "  --coreml             Only build the Core ML backend."
  echo "  --llm                Only build the LLM custom kernels."
  echo "  --mps                Only build the Metal Performance Shaders backend."
  echo "  --optimized          Only build the Optimized kernels."
  echo "  --quantized          Only build the Quantized kernels."
  echo "  --torchao            Only build the TorchAO kernels."
  echo "  --xnnpack            Only build the XNNPACK backend."
  echo
  exit 0
}

CMAKE_OPTIONS_OVERRIDE=()
set_cmake_options_override() {
  local option_name="$1"

  if [[ ${#CMAKE_OPTIONS_OVERRIDE[@]} -eq 0 ]]; then
    # Since the user wants specific options, turn everything off
    CMAKE_OPTIONS_OVERRIDE=(
      "-DEXECUTORCH_BUILD_COREML=OFF"
      "-DEXECUTORCH_BUILD_KERNELS_LLM=OFF"
      "-DEXECUTORCH_BUILD_MPS=OFF"
      "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF"
      "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF"
      "-DEXECUTORCH_BUILD_KERNELS_TORCHAO=OFF"
      "-DEXECUTORCH_BUILD_XNNPACK=OFF"
    )
  fi

  for i in "${!CMAKE_OPTIONS_OVERRIDE[@]}"; do
    if [[ "${CMAKE_OPTIONS_OVERRIDE[$i]}" =~ "-D${option_name}=OFF" ]]; then
      CMAKE_OPTIONS_OVERRIDE[$i]="-D${option_name}=ON"
      break
    fi
  done
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --Release)
        if [[ ! " ${MODES[*]:-} " =~ \bRelease\b ]]; then
          MODES+=("Release")
        fi
        ;;
      --Debug)
        if [[ ! " ${MODES[*]:-} " =~ \bDebug\b ]]; then
          MODES+=("Debug")
        fi
        ;;
      --coreml) set_cmake_options_override "EXECUTORCH_BUILD_COREML";;
      --llm) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_LLM" ;;
      --mps) set_cmake_options_override "EXECUTORCH_BUILD_MPS" ;;
      --optimized) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" ;;
      --quantized) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_QUANTIZED" ;;
      --torchao) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_TORCHAO" ;;
      --xnnpack) set_cmake_options_override "EXECUTORCH_BUILD_XNNPACK" ;;
      *)
        echo -e "\033[31m[error] unknown option: ${arg}\033[0m"
        exit 1
      ;;
  esac
done

# If no modes are specified, default to both Release and Debug
if [[ ${#MODES[@]} -eq 0 ]]; then
  MODES=("Release" "Debug")
fi

echo "Building libraries"

rm -rf "${OUTPUT_DIR}"
for preset_index in "${!PRESETS[@]}"; do
  preset="${PRESETS[$preset_index]}"
  preset_output_dir="${OUTPUT_DIR}/${PRESETS_RELATIVE_OUT_DIR[$preset_index]}"

  for mode in "${MODES[@]}"; do
    echo "Building preset ${preset} (${mode}) in ${preset_output_dir}..."

    # Do NOT add options here. Update the respective presets instead.
    cmake -S "${SOURCE_ROOT_DIR}" \
          -B "${preset_output_dir}" \
          --fresh \
          -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="${preset_output_dir}" \
          -DCMAKE_BUILD_TYPE="${mode}" \
          ${CMAKE_OPTIONS_OVERRIDE[@]:-} \
          --preset "${preset}"

    cmake --build "${preset_output_dir}" \
          --config "${mode}"
  done
done

echo "Exporting headers"

# FRAMEWORK_EXECUTORCH

mkdir -p "$FRAMEWORK_EXECUTORCH_HEADERS_PATH/$FRAMEWORK_EXECUTORCH_MODULE_NAME"

"$SOURCE_ROOT_DIR"/scripts/print_exported_headers.py --buck2=$(realpath "$BUCK2") --targets \
  //extension/module: \
  //extension/tensor: \
| rsync -av --files-from=- "$SOURCE_ROOT_DIR" "$FRAMEWORK_EXECUTORCH_HEADERS_PATH/$FRAMEWORK_EXECUTORCH_MODULE_NAME"

# HACK: XCFrameworks don't appear to support exporting any build
# options, but we need the following:
# - runtime/core/portable/type/c10 reachable with `#include <c10/...>`
# - exported -DC10_USING_CUSTOM_GENERATED_MACROS compiler flag
# So, just patch our generated framework to do that.
sed -i '' '1i\
#define C10_USING_CUSTOM_GENERATED_MACROS
' \
"$FRAMEWORK_EXECUTORCH_HEADERS_PATH/executorch/runtime/core/portable_type/c10/torch/headeronly/macros/Export.h" \
"$FRAMEWORK_EXECUTORCH_HEADERS_PATH/executorch/runtime/core/portable_type/c10/torch/headeronly/macros/Macros.h"

cp -r $FRAMEWORK_EXECUTORCH_HEADERS_PATH/executorch/runtime/core/portable_type/c10/c10 "$FRAMEWORK_EXECUTORCH_HEADERS_PATH/"
cp -r $FRAMEWORK_EXECUTORCH_HEADERS_PATH/executorch/runtime/core/portable_type/c10/torch "$FRAMEWORK_EXECUTORCH_HEADERS_PATH/"

cp "$SOURCE_ROOT_DIR/extension/apple/$FRAMEWORK_EXECUTORCH_MODULE_NAME/Exported/"*.h "$FRAMEWORK_EXECUTORCH_HEADERS_PATH/$FRAMEWORK_EXECUTORCH_MODULE_NAME"

cat > "$FRAMEWORK_EXECUTORCH_HEADERS_PATH/module.modulemap" << EOF
module ${FRAMEWORK_EXECUTORCH_MODULE_NAME} {
  umbrella header "${FRAMEWORK_EXECUTORCH_MODULE_NAME}/${FRAMEWORK_EXECUTORCH_MODULE_NAME}.h"
  export *
}
EOF

# FRAMEWORK_EXECUTORCH_LLM

mkdir -p "$FRAMEWORK_EXECUTORCH_LLM_HEADERS_PATH/$FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME"

cp "$SOURCE_ROOT_DIR/extension/llm/apple/$FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME/Exported/"*.h "$FRAMEWORK_EXECUTORCH_LLM_HEADERS_PATH/$FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME"

cat > "$FRAMEWORK_EXECUTORCH_LLM_HEADERS_PATH/$FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME/module.modulemap" << EOF
module ${FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME} {
  umbrella header "${FRAMEWORK_EXECUTORCH_LLM_MODULE_NAME}.h"
  export *
}
EOF

echo "Creating frameworks"

append_framework_flag() {
  local option_name="$1"
  local framework="$2"
  local mode="$3"

  if [[ ${#CMAKE_OPTIONS_OVERRIDE[@]} -gt 0 && -n "$option_name" ]]; then
    for cmake_option in "${CMAKE_OPTIONS_OVERRIDE[@]}"; do
      if [[ "$cmake_option" =~ "-D${option_name}=OFF" ]]; then
        echo "Skipping framework: ${framework}"
        return
      fi
    done
  fi

  if [[ -n "$mode" && "$mode" != "Release" ]]; then
      local name spec
      name=$(echo "$framework" | cut -d: -f1)
      spec=$(echo "$framework" | cut -d: -f2-)
      framework="${name}_$(echo "$mode" | tr '[:upper:]' '[:lower:]'):${spec}"
  fi
  echo "Adding framework: ${framework}"
  FRAMEWORK_FLAGS+=("--framework=$framework")
}

for mode in "${MODES[@]}"; do
  FRAMEWORK_FLAGS=()
  for preset_out_dir in "${PRESETS_RELATIVE_OUT_DIR[@]}"; do
    echo "Framework directory: ${preset_out_dir}/${mode}"
    FRAMEWORK_FLAGS+=("--directory=${preset_out_dir}/${mode}")
  done

  append_framework_flag "" "$FRAMEWORK_EXECUTORCH" "$mode"
  append_framework_flag "" "$FRAMEWORK_EXECUTORCH_LLM" "$mode"
  append_framework_flag "" "$FRAMEWORK_THREADPOOL" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_COREML" "$FRAMEWORK_BACKEND_COREML" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_MPS" "$FRAMEWORK_BACKEND_MPS" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_XNNPACK" "$FRAMEWORK_BACKEND_XNNPACK" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_LLM" "$FRAMEWORK_KERNELS_LLM" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" "$FRAMEWORK_KERNELS_OPTIMIZED" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_QUANTIZED" "$FRAMEWORK_KERNELS_QUANTIZED" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_TORCHAO" "$FRAMEWORK_KERNELS_TORCHAO" "$mode"

  cd "${OUTPUT_DIR}"
  "$SOURCE_ROOT_DIR"/scripts/create_frameworks.sh "${FRAMEWORK_FLAGS[@]}"
done

echo "Cleaning up"

for preset_out_dir in "${PRESETS_RELATIVE_OUT_DIR[@]}"; do
  rm -rf "${OUTPUT_DIR}/${preset_out_dir}"
done

rm -rf "$FRAMEWORK_EXECUTORCH_HEADERS_PATH"
rm -rf "$FRAMEWORK_EXECUTORCH_LLM_HEADERS_PATH"

echo "Running tests"

cd "$SOURCE_ROOT_DIR"
swift test

echo "Build succeeded!"
