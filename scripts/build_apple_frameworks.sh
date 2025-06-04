#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

MODES=("Release" "Debug")
PRESETS=("ios" "ios-simulator" "macos")
# To support backwards compatibility, we want to retain the same output directory.
PRESETS_RELATIVE_OUT_DIR=("ios" "simulator" "macos")

SOURCE_ROOT_DIR=$(git rev-parse --show-toplevel)
OUTPUT_DIR="${SOURCE_ROOT_DIR}/cmake-out"
HEADERS_RELATIVE_PATH="include"
HEADERS_ABSOLUTE_PATH="${OUTPUT_DIR}/${HEADERS_RELATIVE_PATH}"

BUCK2=$(python3 "$SOURCE_ROOT_DIR/tools/cmake/resolve_buck.py" --cache_dir="$SOURCE_ROOT_DIR/buck2-bin")
if [[ "$BUCK2" == "buck2" ]]; then
  BUCK2=$(command -v buck2)
fi

FRAMEWORK_EXECUTORCH="executorch:\
libexecutorch.a,\
libexecutorch_core.a,\
libextension_apple.a,\
libextension_data_loader.a,\
libextension_flat_tensor.a,\
libextension_module.a,\
libextension_tensor.a,\
:$HEADERS_RELATIVE_PATH:ExecuTorch"

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
libxnnpack_backend.a,\
libmicrokernels-prod.a,\
:"

FRAMEWORK_KERNELS_CUSTOM="kernels_custom:\
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

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Build frameworks for Apple platforms."
  echo
  echo "Options:"
  echo "  --Debug              Build Debug version."
  echo "  --Release            Build Release version."
  echo "  --coreml             Only build the Core ML backend."
  echo "  --custom             Only build the Custom kernels."
  echo "  --mps                Only build the Metal Performance Shaders backend."
  echo "  --optimized          Only build the Optimized kernels."
  echo "  --quantized          Only build the Quantized kernels."
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
      "-DEXECUTORCH_BUILD_KERNELS_CUSTOM=OFF"
      "-DEXECUTORCH_BUILD_MPS=OFF"
      "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF"
      "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF"
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
      --custom) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_CUSTOM" ;;
      --mps) set_cmake_options_override "EXECUTORCH_BUILD_MPS" ;;
      --optimized) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" ;;
      --quantized) set_cmake_options_override "EXECUTORCH_BUILD_KERNELS_QUANTIZED" ;;
      --xnnpack) set_cmake_options_override "EXECUTORCH_BUILD_XNNPACK" ;;
      *)
        echo -e "\033[31m[error] unknown option: ${arg}\033[0m"
        exit 1
      ;;
  esac
done

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
          -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="${preset_output_dir}" \
          -DCMAKE_BUILD_TYPE="${mode}" \
          ${CMAKE_OPTIONS_OVERRIDE[@]:-} \
          --preset "${preset}"

    cmake --build "${preset_output_dir}" \
          --config "${mode}" \
          -j$(sysctl -n hw.ncpu)
  done
done

echo "Exporting headers"

mkdir -p "$HEADERS_ABSOLUTE_PATH"

"$SOURCE_ROOT_DIR"/scripts/print_exported_headers.py --buck2=$(realpath "$BUCK2") --targets \
  //extension/module: \
  //extension/tensor: \
| rsync -av --files-from=- "$SOURCE_ROOT_DIR" "$HEADERS_ABSOLUTE_PATH/executorch"

# HACK: XCFrameworks don't appear to support exporting any build
# options, but we need the following:
# - runtime/core/portable/type/c10 reachable with `#include <c10/...>`
# - exported -DC10_USING_CUSTOM_GENERATED_MACROS compiler flag
# So, just patch our generated framework to do that.
sed -i '' '1i\
#define C10_USING_CUSTOM_GENERATED_MACROS
' \
"$HEADERS_ABSOLUTE_PATH/executorch/runtime/core/portable_type/c10/c10/macros/Macros.h" \
"$HEADERS_ABSOLUTE_PATH/executorch/runtime/core/portable_type/c10/c10/macros/Export.h"

cp -r $HEADERS_ABSOLUTE_PATH/executorch/runtime/core/portable_type/c10/c10 "$HEADERS_ABSOLUTE_PATH/"

cp "$SOURCE_ROOT_DIR/extension/apple/ExecuTorch/Exported/"*.h "$HEADERS_ABSOLUTE_PATH/executorch"

cat > "$HEADERS_ABSOLUTE_PATH/module.modulemap" << 'EOF'
module ExecuTorch {
  umbrella header "ExecuTorch/ExecuTorch.h"
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
  append_framework_flag "" "$FRAMEWORK_THREADPOOL" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_COREML" "$FRAMEWORK_BACKEND_COREML" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_MPS" "$FRAMEWORK_BACKEND_MPS" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_XNNPACK" "$FRAMEWORK_BACKEND_XNNPACK" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_CUSTOM" "$FRAMEWORK_KERNELS_CUSTOM" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" "$FRAMEWORK_KERNELS_OPTIMIZED" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_QUANTIZED" "$FRAMEWORK_KERNELS_QUANTIZED" "$mode"

  cd "${OUTPUT_DIR}"
  "$SOURCE_ROOT_DIR"/scripts/create_frameworks.sh "${FRAMEWORK_FLAGS[@]}"
done

echo "Cleaning up"

for preset_out_dir in "${PRESETS_RELATIVE_OUT_DIR[@]}"; do
  rm -rf "${OUTPUT_DIR}/${preset_out_dir}"
done

rm -rf "$HEADERS_ABSOLUTE_PATH"

echo "Running tests"

cd "$SOURCE_ROOT_DIR"
swift test

echo "Build succeeded!"
