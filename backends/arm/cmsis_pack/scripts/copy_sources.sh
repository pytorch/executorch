#!/bin/bash
# Copy ExecuTorch v1.1.0 sources from Docker container to pack src/
# This script runs INSIDE the Docker container
set -e

EXECUTORCH_SRC="/workspace/executorch"
BUILD_DIR="/workspace2/out"
PACK_SRC="/workspace2/executorch-pack/src"

echo "ExecuTorch source version:"
cd "$EXECUTORCH_SRC" && git describe --tags 2>/dev/null || git log --oneline -1
echo ""

# Clean and recreate pack source directory
rm -rf "${PACK_SRC}"
mkdir -p "${PACK_SRC}"

echo "=== Creating include structure ==="
mkdir -p "${PACK_SRC}/include/executorch"

echo "  Copying runtime sources..."
mkdir -p "${PACK_SRC}/src/runtime"
cp -r "${EXECUTORCH_SRC}/runtime/core" "${PACK_SRC}/src/runtime/"
cp -r "${EXECUTORCH_SRC}/runtime/executor" "${PACK_SRC}/src/runtime/"
cp -r "${EXECUTORCH_SRC}/runtime/kernel" "${PACK_SRC}/src/runtime/"
cp -r "${EXECUTORCH_SRC}/runtime/platform" "${PACK_SRC}/src/runtime/"
cp -r "${EXECUTORCH_SRC}/runtime/backend" "${PACK_SRC}/src/runtime/"

echo "  Copying headers to include path..."
cp -r "${PACK_SRC}/src/runtime" "${PACK_SRC}/include/executorch/"

echo "  Copying kernel sources..."
mkdir -p "${PACK_SRC}/src/kernels"
cp -r "${EXECUTORCH_SRC}/kernels/portable" "${PACK_SRC}/src/kernels/"
cp -r "${EXECUTORCH_SRC}/kernels/quantized" "${PACK_SRC}/src/kernels/"
cp -r "${EXECUTORCH_SRC}/kernels/prim_ops" "${PACK_SRC}/src/kernels/"
cp -r "${PACK_SRC}/src/kernels" "${PACK_SRC}/include/executorch/"

echo "  Copying extension sources..."
mkdir -p "${PACK_SRC}/src/extension"
cp -r "${EXECUTORCH_SRC}/extension/data_loader" "${PACK_SRC}/src/extension/" 2>/dev/null || true
cp -r "${EXECUTORCH_SRC}/extension/memory_allocator" "${PACK_SRC}/src/extension/" 2>/dev/null || true
cp -r "${EXECUTORCH_SRC}/extension/runner_util" "${PACK_SRC}/src/extension/" 2>/dev/null || true
cp -r "${PACK_SRC}/src/extension" "${PACK_SRC}/include/executorch/"

echo "  Copying schema sources..."
mkdir -p "${PACK_SRC}/src/schema"
cp "${EXECUTORCH_SRC}/schema/"*.h "${PACK_SRC}/src/schema/" 2>/dev/null || true
cp "${EXECUTORCH_SRC}/schema/"*.cpp "${PACK_SRC}/src/schema/" 2>/dev/null || true
cp "${EXECUTORCH_SRC}/schema/"*.fbs "${PACK_SRC}/src/schema/" 2>/dev/null || true

echo "  Copying generated schema headers..."
if [ -d "${BUILD_DIR}/stage1/schema/include/executorch/schema" ]; then
    cp "${BUILD_DIR}/stage1/schema/include/executorch/schema/"*.h "${PACK_SRC}/src/schema/" 2>/dev/null || true
fi
cp -r "${PACK_SRC}/src/schema" "${PACK_SRC}/include/executorch/"

echo "  Copying generated headers (flatbuffers)..."
if [ -d "${BUILD_DIR}/stage1/third-party/flatc_ep/include" ]; then
    cp -r "${BUILD_DIR}/stage1/third-party/flatc_ep/include/flatbuffers" "${PACK_SRC}/include/"
fi

echo "  Creating c10 and torch include paths..."
C10_DIR="${PACK_SRC}/src/runtime/core/portable_type/c10/c10"
TORCH_DIR="${PACK_SRC}/src/runtime/core/portable_type/c10/torch"
if [ -d "${C10_DIR}" ]; then
    cp -r "${C10_DIR}" "${PACK_SRC}/include/"
fi
if [ -d "${TORCH_DIR}" ]; then
    cp -r "${TORCH_DIR}" "${PACK_SRC}/include/"
fi

echo "  Copying backend sources..."
mkdir -p "${PACK_SRC}/src/backends"
if [ -d "${EXECUTORCH_SRC}/backends/arm" ]; then
    cp -r "${EXECUTORCH_SRC}/backends/arm" "${PACK_SRC}/src/backends/"
fi
if [ -d "${EXECUTORCH_SRC}/backends/cortex_m" ]; then
    cp -r "${EXECUTORCH_SRC}/backends/cortex_m" "${PACK_SRC}/src/backends/"
fi
cp -r "${PACK_SRC}/src/backends" "${PACK_SRC}/include/executorch/"

echo "  Copying generated registration files..."
mkdir -p "${PACK_SRC}/generated"
if [ -d "${BUILD_DIR}/stage2/executorch_selected_kernels" ]; then
    cp -r "${BUILD_DIR}/stage2/executorch_selected_kernels" "${PACK_SRC}/generated/"
fi

echo "  Copying kernel registration module..."
mkdir -p "${PACK_SRC}/src/registration"
if [ -f "/workspace2/executorch-pack/output/src/registration/RegisterAllKernels.cpp" ]; then
    cp "/workspace2/executorch-pack/output/src/registration/RegisterAllKernels.cpp" "${PACK_SRC}/src/registration/"
    echo "    Copied RegisterAllKernels.cpp from executorch-pack/output"
fi

echo "  Copying operator metadata..."
if [ -f "${BUILD_DIR}/stage2/assets/meta/selected_operators.yaml" ]; then
    mkdir -p "${PACK_SRC}/meta"
    cp "${BUILD_DIR}/stage2/assets/meta/selected_operators.yaml" "${PACK_SRC}/meta/"
fi

echo "  Copying stubs..."
mkdir -p "${PACK_SRC}/src/stubs"
if [ -f "/workspace2/src/posix_stub.cpp" ]; then
    cp "/workspace2/src/posix_stub.cpp" "${PACK_SRC}/src/stubs/"
fi
if [ -f "/workspace2/src/random_ops_stubs.cpp" ]; then
    cp "/workspace2/src/random_ops_stubs.cpp" "${PACK_SRC}/src/stubs/"
fi

echo ""
echo "=== Source copy complete ==="
echo "Total .cpp files: $(find "${PACK_SRC}" -name '*.cpp' | wc -l)"
echo "Total .h files: $(find "${PACK_SRC}" -name '*.h' | wc -l)"
