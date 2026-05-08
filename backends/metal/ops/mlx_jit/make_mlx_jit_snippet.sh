#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from MLX 0.31.2's make_compiled_preamble.sh
#   (executorch/backends/mlx/third-party/mlx/mlx/backend/metal/
#    make_compiled_preamble.sh)
#
# Generates a C++ source file exposing the JIT preamble of an MLX MSL header
# as a free function in our `executorch::backends::metal_v2::mlx_jit::Snippets`
# namespace.
#
# Usage:
#   make_mlx_jit_snippet.sh <output_dir> <mlx_src_dir> <src_subpath> [src_path]
#
#   <output_dir>  : directory to write <name>.cpp into
#   <mlx_src_dir> : path to MLX submodule root (above mlx/backend/metal)
#   <src_subpath> : subpath of header relative to kernels/, without .h —
#                   used to derive the function name (basename) and, when
#                   <src_path> is omitted, the input file path as
#                   ${mlx_src_dir}/mlx/backend/metal/kernels/${src_subpath}.h
#   <src_path>    : optional. Absolute path to the input file. Use this to
#                   vendor a snippet from a non-MLX-submodule location
#                   (e.g. ops/mlx_jit/local_snippets/gemv.h which thinly
#                   wraps gemv.metal with the AOT-instantiation macros
#                   redefined to nothing). The file may be .h, .metal, or
#                   any other extension xcrun metal -E accepts.
#
# The function name is derived from basename(src_subpath); e.g.
#   steel/gemm/gemm                            -> gemm()
#   steel/gemm/kernels/steel_gemm_fused_nax    -> steel_gemm_fused_nax()
#   gemv (with src_path = local_snippets/gemv.h) -> gemv()
#
# All snippets live in OUR namespace and can be referenced from
# Snippets.h.
set -eo pipefail

OUTPUT_DIR=$1
SRC_DIR=$2
SRC_FILE=$3
SRC_PATH=${4:-${SRC_DIR}/mlx/backend/metal/kernels/${SRC_FILE}.h}
SRC_NAME=$(basename -- "${SRC_FILE}")
JIT_INCLUDES=${SRC_DIR}/mlx/backend/metal/kernels/jit
INPUT_FILE=${SRC_PATH}
OUTPUT_FILE=${OUTPUT_DIR}/${SRC_NAME}.cpp

mkdir -p "$OUTPUT_DIR"

CCC="xcrun -sdk macosx metal -x metal"
HDRS=$( $CCC -I"$SRC_DIR" -I"$JIT_INCLUDES" -DMLX_METAL_JIT -E -P -CC -C -H "$INPUT_FILE" -w 2>&1 1>/dev/null )

if [ -n "$HDRS" ]; then
  invalid_lines=$(echo "$HDRS" | grep -v '^\.*\.' || true)
  if [ -n "$invalid_lines" ]; then
    echo "Error: Metal compiler header resolution failed for ${INPUT_FILE}" >&2
    echo "Expected lines starting with '.' but got:" >&2
    echo "$invalid_lines" >&2
    exit 1
  fi
fi

# Strip Xcode toolchain headers AND Metal SDK system headers (under
# /private/.../com.apple.MobileAsset.MetalToolchain-* — system stdlib that
# the runtime Metal compiler will resolve via #include <metal_stdlib>).
HDRS=$(echo "$HDRS" | grep -v "Xcode" | grep -v "MobileAsset.MetalToolchain")

declare -a HDRS_LIST=($HDRS)
declare -a HDRS_STACK=()
declare -a HDRS_SORTED=()

length=${#HDRS_LIST[@]}
HDRS_LIST+=(".")

for ((i=0; i<${length}; i+=2)); do
  header="${HDRS_LIST[$i+1]#$SRC_DIR/}"

  # Skip absolute paths that survived the system-header filter (defensive).
  case "$header" in
    /*) continue ;;
  esac

  str_this="${HDRS_LIST[$i]}"
  str_next="${HDRS_LIST[$i + 2]}"
  depth_this=${#str_this}
  depth_next=${#str_next}

  if [ $depth_next -gt $depth_this ]; then
    HDRS_STACK=($header ${HDRS_STACK[@]})
  else
    HDRS_SORTED+=($header)
    pop_len=$((depth_this - depth_next))
    for popped_header in "${HDRS_STACK[@]:0:$pop_len}"; do
      HDRS_SORTED+=($popped_header)
    done
    HDRS_STACK=(${HDRS_STACK[@]:$pop_len})
  fi
done

HDRS_SORTED+=("${INPUT_FILE#$SRC_DIR/}")

CONTENT=$(
echo "// Copyright © 2025 Apple Inc."
echo ""
echo "// Auto generated source for ${INPUT_FILE#$SRC_DIR/}"
echo ""

for header in "${HDRS_SORTED[@]}"; do
  # Defensive: skip if the file doesn't exist (system header survived).
  if [ ! -f "${SRC_DIR}/${header}" ]; then
    continue
  fi
  echo "///////////////////////////////////////////////////////////////////////////////"
  echo "// Contents from \"${header}\""
  echo "///////////////////////////////////////////////////////////////////////////////"
  echo ""
  echo "#line 1 \"${header}\""
  grep -h -v -G -e "#include \".*.h\"" -e "#pragma once" "${SRC_DIR}/${header}"
  echo ""
done

echo "///////////////////////////////////////////////////////////////////////////////"
)

# Strip MLX's AOT instantiation macros AND their invocations. The xcrun
# metal preprocessor preserves `#define` directives + macro calls in the
# output text; the runtime metal compiler then re-evaluates them and
# generates AOT [[host_name(...)]] template instantiations that collide
# with our per-shape JIT instantiation.
#
# IMPORTANT: only strip the AOT-KERNEL instantiation macros (those that
# generate `template [[host_name(...)]]` entries). Other `instantiate_*`
# macros in utils.h (instantiate_default_limit, instantiate_metal_math_funcs,
# instantiate_metal_simd_*) generate type-specialization structs that
# OTHER kernels need — stripping them breaks Limits<float> etc.
#
# Specifically strip:
#   - `#define instantiate_kernel(...)` and its multi-line continuations
#   - `#define instantiate_gemv*(...)`
#   - `#define instantiate_sdpa_vector*(...)`  (SDPA vector kernels)
#   - `#define instantiate_attn*(...)`         (steel SDPA / steel-NAX SDPA)
#   - `#define instantiate_quantized*(...)`    (affine-quantized linear)
#   - top-level `instantiate_kernel(...);`
#   - top-level `instantiate_gemv*(...);`
#   - top-level `instantiate_sdpa_vector*(...);`
#   - top-level `instantiate_attn*(...);`
#   - top-level `instantiate_quantized*(...);`
#
# NOTE: `instantiate_quantized*` matches the full family in
# kernels/quantized.metal (instantiate_quantized, _batched, _aligned,
# _quad, _split_k, _all_*, _splitk_qmm, _all_rhs, _funcs, _types, _groups,
# _all). It does NOT collide with utils.h's `instantiate_*` (those start
# with `instantiate_default_limit`, `instantiate_metal_*`, etc.).
CONTENT=$(echo "$CONTENT" | awk '
  BEGIN { in_define = 0; in_call = 0 }
  in_define {
    if ($0 !~ /\\[[:space:]]*$/) { in_define = 0 }
    next
  }
  in_call {
    if ($0 ~ /;/) { in_call = 0 }
    next
  }
  /^[[:space:]]*#define[[:space:]]+(instantiate_kernel|instantiate_gemv|instantiate_sdpa_vector|instantiate_attn|instantiate_quantized)/ {
    if ($0 ~ /\\[[:space:]]*$/) { in_define = 1 }
    next
  }
  /^[[:space:]]*(instantiate_kernel|instantiate_gemv|instantiate_sdpa_vector|instantiate_attn|instantiate_quantized)[a-zA-Z_]*[[:space:]]*\(/ {
    if ($0 !~ /;/) { in_call = 1 }
    next
  }
  { print }
')

cat << EOF > "$OUTPUT_FILE"
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Auto-generated by make_mlx_jit_snippet.sh from MLX 0.31.2 submodule.
 * DO NOT EDIT — regenerated at every build.
 */

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace mlx_jit {
namespace Snippets {

const char* ${SRC_NAME}() {
  return R"MLX_JIT_PREAMBLE(
$CONTENT
)MLX_JIT_PREAMBLE";
}

} // namespace Snippets
} // namespace mlx_jit
} // namespace metal_v2
} // namespace backends
} // namespace executorch
EOF
