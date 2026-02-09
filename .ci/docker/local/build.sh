#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Local Docker build script for development/hackathon use.
# This script patches the CI Docker build to skip sccache entirely.
#
# Usage:
#   ./build.sh executorch-ubuntu-22.04-clang12 -t executorch-hackathon:basic
#   ./build.sh executorch-ubuntu-22.04-clang12-android -t executorch-hackathon:android

set -exu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "${SCRIPT_DIR}")"

# Create temporary patched files
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

# Copy the common directory to temp
cp -r "${DOCKER_DIR}/common" "${TEMP_DIR}/common"

# Replace install_cache.sh with a no-op version (skip sccache entirely)
cat > "${TEMP_DIR}/common/install_cache.sh" << 'EOF'
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Local build version - skip sccache entirely, use compilers directly.
# sccache is only useful for CI where builds are cached across runs.

echo "Skipping sccache installation (local build)"
EOF

# Copy ubuntu directory to temp and patch Dockerfile
cp -r "${DOCKER_DIR}/ubuntu" "${TEMP_DIR}/ubuntu"

# Remove sccache-related ENV and PATH modifications from Dockerfile
sed -i.bak \
  -e 's/^ENV SCCACHE_BUCKET/#ENV SCCACHE_BUCKET/' \
  -e 's/^ENV SCCACHE_S3_KEY_PREFIX/#ENV SCCACHE_S3_KEY_PREFIX/' \
  -e 's/^ENV SCCACHE_REGION/#ENV SCCACHE_REGION/' \
  -e 's|^ENV PATH /opt/cache/bin|#ENV PATH /opt/cache/bin|' \
  "${TEMP_DIR}/ubuntu/Dockerfile"
rm -f "${TEMP_DIR}/ubuntu/Dockerfile.bak"

# Copy other necessary files
cp "${DOCKER_DIR}/requirements-ci.txt" "${TEMP_DIR}/" 2>/dev/null || true
cp "${DOCKER_DIR}/conda-env-ci.txt" "${TEMP_DIR}/" 2>/dev/null || true
cp "${DOCKER_DIR}/../../ci_commit_pins/pytorch.txt" "${TEMP_DIR}/pytorch.txt" 2>/dev/null || true
cp "${DOCKER_DIR}/../../ci_commit_pins/buck2.txt" "${TEMP_DIR}/buck2.txt" 2>/dev/null || true
cp "${DOCKER_DIR}/../../requirements-lintrunner.txt" "${TEMP_DIR}/" 2>/dev/null || true
cp -r "${DOCKER_DIR}/../../examples/arm" "${TEMP_DIR}/arm" 2>/dev/null || true

# Now run the build from temp directory with patched files
cd "${TEMP_DIR}"

FULL_IMAGE_NAME="$1"
shift

IMAGE_NAME=$(echo "${FULL_IMAGE_NAME}" | sed 's/ci-image://')

echo "Building ${IMAGE_NAME} Docker image (local build - no sccache)"

OS=ubuntu
OS_VERSION=22.04
CLANG_VERSION=""
GCC_VERSION=""
PYTHON_VERSION=3.10
MINICONDA_VERSION=23.10.0-1
BUCK2_VERSION=$(cat buck2.txt 2>/dev/null || echo "latest")

case "${IMAGE_NAME}" in
  executorch-ubuntu-22.04-gcc11)
    LINTRUNNER=""
    GCC_VERSION=11
    ;;
  executorch-ubuntu-22.04-clang12)
    LINTRUNNER=""
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-clang12-android)
    LINTRUNNER=""
    CLANG_VERSION=12
    ANDROID_NDK_VERSION=r28c
    ;;
  executorch-ubuntu-22.04-arm-sdk)
    ARM_SDK=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-qnn-sdk)
    QNN_SDK=yes
    CLANG_VERSION=12
    ;;
  executorch-ubuntu-22.04-mediatek-sdk)
    MEDIATEK_SDK=yes
    CLANG_VERSION=12
    ANDROID_NDK_VERSION=r28c
    ;;
  *)
    echo "Invalid image name ${IMAGE_NAME}"
    echo "Supported images:"
    echo "  executorch-ubuntu-22.04-gcc11"
    echo "  executorch-ubuntu-22.04-clang12"
    echo "  executorch-ubuntu-22.04-clang12-android"
    echo "  executorch-ubuntu-22.04-arm-sdk"
    echo "  executorch-ubuntu-22.04-qnn-sdk"
    echo "  executorch-ubuntu-22.04-mediatek-sdk"
    exit 1
esac

TORCH_VERSION=$(cat pytorch.txt 2>/dev/null || echo "main")
BUILD_DOCS=1

docker build \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "GCC_VERSION=${GCC_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "BUCK2_VERSION=${BUCK2_VERSION}" \
  --build-arg "LINTRUNNER=${LINTRUNNER:-}" \
  --build-arg "BUILD_DOCS=${BUILD_DOCS}" \
  --build-arg "ARM_SDK=${ARM_SDK:-}" \
  --build-arg "QNN_SDK=${QNN_SDK:-}" \
  --build-arg "MEDIATEK_SDK=${MEDIATEK_SDK:-}" \
  --build-arg "ANDROID_NDK_VERSION=${ANDROID_NDK_VERSION:-}" \
  -f "ubuntu/Dockerfile" \
  "$@" \
  .
