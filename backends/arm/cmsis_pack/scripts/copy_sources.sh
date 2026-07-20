#!/usr/bin/env bash
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copy ExecuTorch sources from the repo tree and CMake build outputs
# into a flat pack staging directory.
#
# Only .cpp, .h, .c, .fbs, and .yaml files are kept.  Test directories,
# Python files, build-system files, and other non-essential assets are
# stripped to match the curated set produced by the legacy Docker workflow.
#
# Usage:
#   ./copy_sources.sh --executorch-root <path> --build-dir <path> \
#                     --pack-staging <path>
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_DIR="$(cd "$SCRIPT_DIR/../../cmsis_pack" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
EXECUTORCH_ROOT=""
BUILD_DIR=""
PACK_SRC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --executorch-root) EXECUTORCH_ROOT="$2"; shift 2 ;;
        --build-dir)      BUILD_DIR="$2";        shift 2 ;;
        --pack-staging)   PACK_SRC="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXECUTORCH_ROOT" || -z "$BUILD_DIR" || -z "$PACK_SRC" ]]; then
    echo "Usage: $0 --executorch-root <path> --build-dir <path> --pack-staging <path>"
    exit 1
fi

echo "ExecuTorch root: $EXECUTORCH_ROOT"
echo "CMake build dir: $BUILD_DIR"
echo "Pack staging:    $PACK_SRC"
echo ""

# ---------------------------------------------------------------------------
# Helper: strip non-essential files after a bulk cp -r
# Removes test dirs, Python, build-system files, __pycache__, etc.
# ---------------------------------------------------------------------------
strip_non_essential() {
    local dir="$1"
    # Remove test directories
    find "$dir" -type d \( -name 'test' -o -name 'tests' -o -name 'testing_util' \
        -o -name '__pycache__' \) -exec rm -rf {} + 2>/dev/null || true
    # Remove non-source files
    find "$dir" -type f \( \
        -name '*.py' -o -name '*.pyc' -o -name '*.pyi' \
        -o -name '*.sh' -o -name '*.bat' \
        -o -name '*.bzl' -o -name 'TARGETS' -o -name 'BUCK' \
        -o -name 'CMakeLists.txt' -o -name '*.cmake' \
        -o -name 'BUILD' -o -name 'BUILD.bazel' \
        -o -name '*.md' -o -name '*.rst' -o -name '*.txt' \
        -o -name '*.toml' -o -name '*.cfg' -o -name '*.ini' \
        -o -name '*.json' -o -name '*.lock' \
        -o -name '.gitignore' -o -name '.clang-format' \
        -o -name 'Makefile' \
    \) -delete 2>/dev/null || true
    # Remove *_test.cpp files
    find "$dir" -type f -name '*_test.cpp' -delete 2>/dev/null || true
    # Remove empty directories left behind
    find "$dir" -type d -empty -delete 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Create include / src structure
# ---------------------------------------------------------------------------
mkdir -p "${PACK_SRC}/include/executorch"

echo "  Copying runtime sources..."
mkdir -p "${PACK_SRC}/src/runtime"
for d in core executor kernel platform backend; do
    if [[ -d "${EXECUTORCH_ROOT}/runtime/$d" ]]; then
        cp -r "${EXECUTORCH_ROOT}/runtime/$d" "${PACK_SRC}/src/runtime/"
    fi
done
strip_non_essential "${PACK_SRC}/src/runtime"

# Apply pack-local patch to runtime/platform/default/minimal.cpp.
# Marks each et_pal_* fallback ET_WEAK on its definition; required for
# GCC 13/14/15 and armclang 6.24 (see the patch header for details).
# Using `patch` rather than overlaying a full pristine copy makes the
# delta self-documenting and surfaces conflicts loudly if upstream
# minimal.cpp changes.
MINIMAL_PATCH="${PACK_DIR}/contributions/runtime/platform/default/minimal.cpp.patch"
if [[ ! -f "$MINIMAL_PATCH" ]]; then
    echo "ERROR: minimal.cpp.patch not found at $MINIMAL_PATCH" >&2
    exit 1
fi
patch --quiet --strip=1 --directory="${PACK_SRC}/src" \
      --input="$MINIMAL_PATCH"

cp -r "${PACK_SRC}/src/runtime" "${PACK_SRC}/include/executorch/"

echo "  Copying kernel sources..."
mkdir -p "${PACK_SRC}/src/kernels"
for d in portable quantized prim_ops; do
    if [[ -d "${EXECUTORCH_ROOT}/kernels/$d" ]]; then
        cp -r "${EXECUTORCH_ROOT}/kernels/$d" "${PACK_SRC}/src/kernels/"
    fi
done
strip_non_essential "${PACK_SRC}/src/kernels"
cp -r "${PACK_SRC}/src/kernels" "${PACK_SRC}/include/executorch/"

echo "  Copying extension sources..."
mkdir -p "${PACK_SRC}/src/extension"
for d in data_loader memory_allocator runner_util; do
    if [[ -d "${EXECUTORCH_ROOT}/extension/$d" ]]; then
        cp -r "${EXECUTORCH_ROOT}/extension/$d" "${PACK_SRC}/src/extension/"
    fi
done
strip_non_essential "${PACK_SRC}/src/extension"
cp -r "${PACK_SRC}/src/extension" "${PACK_SRC}/include/executorch/"

echo "  Copying schema sources..."
mkdir -p "${PACK_SRC}/src/schema"
cp "${EXECUTORCH_ROOT}/schema/"*.h   "${PACK_SRC}/src/schema/" 2>/dev/null || true
cp "${EXECUTORCH_ROOT}/schema/"*.cpp "${PACK_SRC}/src/schema/" 2>/dev/null || true
cp "${EXECUTORCH_ROOT}/schema/"*.fbs "${PACK_SRC}/src/schema/" 2>/dev/null || true

# Generated schema headers from CMake build
for candidate in \
    "${BUILD_DIR}/schema/include/executorch/schema" \
    "${BUILD_DIR}/stage1/schema/include/executorch/schema"; do
    if [[ -d "$candidate" ]]; then
        cp "$candidate/"*.h "${PACK_SRC}/src/schema/" 2>/dev/null || true
        break
    fi
done
cp -r "${PACK_SRC}/src/schema" "${PACK_SRC}/include/executorch/"

echo "  Copying generated flatbuffers headers..."
for candidate in \
    "${BUILD_DIR}/third-party/flatbuffers/include/flatbuffers" \
    "${BUILD_DIR}/third-party/flatc_ep/include/flatbuffers" \
    "${BUILD_DIR}/stage1/third-party/flatc_ep/include/flatbuffers"; do
    if [[ -d "$candidate" ]]; then
        cp -r "$candidate" "${PACK_SRC}/include/"
        break
    fi
done

# Flatbuffers headers are redistributed inside the pack; ship the
# upstream Apache-2.0 LICENSE alongside per its terms.
if [[ -f "${EXECUTORCH_ROOT}/third-party/flatbuffers/LICENSE" ]] \
   && [[ -d "${PACK_SRC}/include/flatbuffers" ]]; then
    cp "${EXECUTORCH_ROOT}/third-party/flatbuffers/LICENSE" \
       "${PACK_SRC}/include/flatbuffers/LICENSE"
fi

echo "  Creating c10 and torch include paths..."
C10_DIR="${PACK_SRC}/src/runtime/core/portable_type/c10/c10"
TORCH_DIR="${PACK_SRC}/src/runtime/core/portable_type/c10/torch"
[[ -d "$C10_DIR" ]]   && cp -r "$C10_DIR"   "${PACK_SRC}/include/"
[[ -d "$TORCH_DIR" ]] && cp -r "$TORCH_DIR" "${PACK_SRC}/include/"

echo "  Copying backend sources..."
mkdir -p "${PACK_SRC}/src/backends/arm"
mkdir -p "${PACK_SRC}/src/backends/cortex_m"

# Arm backend: only the runtime directory (EthosUBackend, VelaBinStream)
if [[ -d "${EXECUTORCH_ROOT}/backends/arm/runtime" ]]; then
    cp -r "${EXECUTORCH_ROOT}/backends/arm/runtime" "${PACK_SRC}/src/backends/arm/"
fi

# Cortex-M backend: ops and cortex_m_ops_lib
for d in ops cortex_m_ops_lib; do
    if [[ -d "${EXECUTORCH_ROOT}/backends/cortex_m/$d" ]]; then
        cp -r "${EXECUTORCH_ROOT}/backends/cortex_m/$d" "${PACK_SRC}/src/backends/cortex_m/"
    fi
done

strip_non_essential "${PACK_SRC}/src/backends"
cp -r "${PACK_SRC}/src/backends" "${PACK_SRC}/include/executorch/"


echo "  Preparing registration directory..."
mkdir -p "${PACK_SRC}/src/registration"

echo ""
echo "=== Source copy complete ==="
echo "Total .cpp files: $(find "${PACK_SRC}" -name '*.cpp' | wc -l)"
echo "Total .h files:   $(find "${PACK_SRC}" -name '*.h'   | wc -l)"
