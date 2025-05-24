#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

MODES=("Release" "Debug")
PRESETS=("ios" "ios-simulator" "macos")

SOURCE_ROOT_DIR=$(git rev-parse --show-toplevel)
OUTPUT_DIR="${SOURCE_ROOT_DIR}/cmake-out"
HEADERS_RELATIVE_PATH="include"
HEADERS_ABSOLUTE_PATH="${OUTPUT_DIR}/${HEADERS_RELATIVE_PATH}"

FRAMEWORK_EXECUTORCH="executorch:\
libexecutorch.a,\
libexecutorch_core.a,\
libextension_apple.a,\
libextension_data_loader.a,\
libextension_flat_tensor.a,\
libextension_module.a,\
libextension_tensor.a,\
:$HEADERS_RELATIVE_PATH:ExecuTorch"

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
libcpuinfo.a,\
libextension_threadpool.a,\
libpthreadpool.a,\
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
:"

FRAMEWORK_KERNELS_PORTABLE="kernels_portable:\
libportable_kernels.a,\
libportable_ops_lib.a,\
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
  echo
  exit 0
}

deprecated_option() {
  local option_name="$1"
  local cmake_var="${2:-}"
  if [[ -n "$cmake_var" ]]; then
    echo -e "\033[31m[error] Flag '--${option_name}' is now ON by default and deprecated. To turn off that feature, use:\033[34m\n\n\tCMAKE_ARGS=\"-D${cmake_var}=OFF\" $0\n\033[0m"
  else
    echo -e "\033[31m[error] Flag '--${option_name}' is now ON by default and deprecated.\033[0m"
  fi
  exit 1
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
      --coreml) deprecated_option "coreml" "EXECUTORCH_BUILD_COREML";;
      --custom) deprecated_option "custom" "EXECUTORCH_BUILD_KERNELS_CUSTOM" ;;
      --mps) deprecated_option "mps" "EXECUTORCH_BUILD_MPS" ;;
      --optimized) deprecated_option "optimized" "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" ;;
      --portable) deprecated_option "portable" ;;
      --quantized) deprecated_option "quantized" "EXECUTORCH_BUILD_KERNELS_QUANTIZED" ;;
      --xnnpack) deprecated_option "xnnpack" "EXECUTORCH_BUILD_XNNPACK" ;;
      *)
      ;;
  esac
done

BUCK2=$(python3 "$SOURCE_ROOT_DIR/tools/cmake/resolve_buck.py" --cache_dir="$SOURCE_ROOT_DIR/buck2-bin")

if [[ "$BUCK2" == "buck2" ]]; then
  BUCK2=$(command -v buck2)
fi

echo "Building libraries"

rm -rf "${OUTPUT_DIR}"
for preset in "${PRESETS[@]}"; do
  for mode in "${MODES[@]}"; do
    output_dir="${OUTPUT_DIR}/${preset}"
    echo "Building preset ${preset} (${mode}) in ${output_dir}..."

    # Do NOT add options here. Update the respective presets instead.
    cmake -S "${SOURCE_ROOT_DIR}" \
          -B "${output_dir}" \
          -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="${output_dir}" \
          -DCMAKE_BUILD_TYPE="${mode}" \
          ${CMAKE_ARGS:-} \
          --preset "${preset}"

    cmake --build "${output_dir}" \
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
  local option="$1"
  local framework="$2"
  local mode="$3"

  if [[ "${CMAKE_ARGS:-}" =~ "-D${option}=OFF" ]]; then
    echo "Skipping framework: ${framework}"
    return
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
  for preset in "${PRESETS[@]}"; do
    echo "Framework directory: ${preset}/${mode}"
    FRAMEWORK_FLAGS+=("--directory=${preset}/${mode}")
  done

  append_framework_flag "" "$FRAMEWORK_EXECUTORCH" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_COREML" "$FRAMEWORK_BACKEND_COREML" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_MPS" "$FRAMEWORK_BACKEND_MPS" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_XNNPACK" "$FRAMEWORK_BACKEND_XNNPACK" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_CUSTOM" "$FRAMEWORK_KERNELS_CUSTOM" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_OPTIMIZED" "$FRAMEWORK_KERNELS_OPTIMIZED" "$mode"
  append_framework_flag "" "$FRAMEWORK_KERNELS_PORTABLE" "$mode"
  append_framework_flag "EXECUTORCH_BUILD_KERNELS_QUANTIZED" "$FRAMEWORK_KERNELS_QUANTIZED" "$mode"

  cd "${OUTPUT_DIR}"
  "$SOURCE_ROOT_DIR"/scripts/create_frameworks.sh "${FRAMEWORK_FLAGS[@]}"
done

echo "Cleaning up"

for preset in "${PRESETS[@]}"; do
  rm -rf "${OUTPUT_DIR}/${preset}/$preset"
done

rm -rf "$HEADERS_ABSOLUTE_PATH"

echo "Running tests"

cd "$SOURCE_ROOT_DIR"
swift test

echo "Build succeeded!"
