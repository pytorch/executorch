#!/bin/bash
# Copy ExecuTorch sources from the repo tree and CMake build outputs
# into a flat pack staging directory.
#
# Usage:
#   ./copy_sources.sh --executorch-root <path> --build-dir <path> \
#                     --pack-staging <path>
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_DIR="$(dirname "$SCRIPT_DIR")"

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
cp -r "${PACK_SRC}/src/runtime" "${PACK_SRC}/include/executorch/"

echo "  Copying kernel sources..."
mkdir -p "${PACK_SRC}/src/kernels"
for d in portable quantized prim_ops; do
    if [[ -d "${EXECUTORCH_ROOT}/kernels/$d" ]]; then
        cp -r "${EXECUTORCH_ROOT}/kernels/$d" "${PACK_SRC}/src/kernels/"
    fi
done
cp -r "${PACK_SRC}/src/kernels" "${PACK_SRC}/include/executorch/"

echo "  Copying extension sources..."
mkdir -p "${PACK_SRC}/src/extension"
for d in data_loader memory_allocator runner_util; do
    if [[ -d "${EXECUTORCH_ROOT}/extension/$d" ]]; then
        cp -r "${EXECUTORCH_ROOT}/extension/$d" "${PACK_SRC}/src/extension/"
    fi
done
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

echo "  Creating c10 and torch include paths..."
C10_DIR="${PACK_SRC}/src/runtime/core/portable_type/c10/c10"
TORCH_DIR="${PACK_SRC}/src/runtime/core/portable_type/c10/torch"
[[ -d "$C10_DIR" ]]   && cp -r "$C10_DIR"   "${PACK_SRC}/include/"
[[ -d "$TORCH_DIR" ]] && cp -r "$TORCH_DIR" "${PACK_SRC}/include/"

echo "  Copying backend sources..."
mkdir -p "${PACK_SRC}/src/backends"
for backend in arm cortex_m; do
    if [[ -d "${EXECUTORCH_ROOT}/backends/$backend" ]]; then
        cp -r "${EXECUTORCH_ROOT}/backends/$backend" "${PACK_SRC}/src/backends/"
    fi
done
cp -r "${PACK_SRC}/src/backends" "${PACK_SRC}/include/executorch/"

echo "  Copying stubs..."
mkdir -p "${PACK_SRC}/src/stubs"
if [[ -d "$PACK_DIR/stubs" ]]; then
    cp "$PACK_DIR/stubs/"*.cpp "${PACK_SRC}/src/stubs/" 2>/dev/null || true
fi

echo "  Preparing registration directory..."
mkdir -p "${PACK_SRC}/src/registration"

echo ""
echo "=== Source copy complete ==="
echo "Total .cpp files: $(find "${PACK_SRC}" -name '*.cpp' | wc -l)"
echo "Total .h files:   $(find "${PACK_SRC}" -name '*.h'   | wc -l)"
