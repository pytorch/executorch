#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Local Docker build script for development/hackathon use.
# This script patches the CI Docker build to work without S3 sccache.
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

# Copy the common directory to temp and patch install_cache.sh
cp -r "${DOCKER_DIR}/common" "${TEMP_DIR}/common"

# Patch install_cache.sh to use local cache instead of S3
cat > "${TEMP_DIR}/common/install_cache.sh" << 'EOF'
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Local build version - uses local sccache instead of S3

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_ubuntu() {
  echo "Preparing to build sccache from source"
  apt-get update
  apt-get install -y cargo
  echo "Checking out sccache repo"
  git clone https://github.com/mozilla/sccache -b v0.8.2

  cd sccache
  echo "Building sccache"
  cargo build --release
  cp target/release/sccache /opt/cache/bin
  echo "Cleaning up"
  cd ..
  rm -rf sccache
  apt-get remove -y cargo rustc
  apt-get autoclean && apt-get clean
}

mkdir -p /opt/cache/bin
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

install_ubuntu

function write_sccache_stub() {
  BINARY=$1
  if [ $1 == "gcc" ]; then
    cat >"/opt/cache/bin/$1" <<STUB
#!/bin/sh
if [ "\$1" = "-E" ] || [ "\$2" = "-E" ]; then
  exec $(which $1) "\$@"
elif [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which $1) "\$@"
else
  exec $(which $1) "\$@"
fi
STUB
  else
    cat >"/opt/cache/bin/$1" <<STUB
#!/bin/sh
if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which $1) "\$@"
else
  exec $(which $1) "\$@"
fi
STUB
  fi
  chmod a+x "/opt/cache/bin/${BINARY}"
}

init_sccache() {
  # Use local cache instead of S3
  export SCCACHE_DIR=/tmp/sccache
  export SCCACHE_CACHE_SIZE=10G
  export SCCACHE_IDLE_TIMEOUT=0
  export SCCACHE_ERROR_LOG=/tmp/sccache_error.log
  export RUST_LOG=sccache::server=error

  as_ci_user sccache --stop-server >/dev/null 2>&1 || true
  rm -f "${SCCACHE_ERROR_LOG}" || true
  as_ci_user sccache --zero-stats || true
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++
init_sccache
EOF

# Copy ubuntu directory to temp and patch Dockerfile
cp -r "${DOCKER_DIR}/ubuntu" "${TEMP_DIR}/ubuntu"

# Patch Dockerfile to remove S3 sccache config
sed -i.bak \
  -e 's/^ENV SCCACHE_BUCKET/#ENV SCCACHE_BUCKET/' \
  -e 's/^ENV SCCACHE_S3_KEY_PREFIX/#ENV SCCACHE_S3_KEY_PREFIX/' \
  -e 's/^ENV SCCACHE_REGION/#ENV SCCACHE_REGION/' \
  "${TEMP_DIR}/ubuntu/Dockerfile"
rm -f "${TEMP_DIR}/ubuntu/Dockerfile.bak"

# Add local sccache config to Dockerfile
sed -i.bak \
  '/^#ENV SCCACHE_REGION/a\
ENV SCCACHE_DIR=/tmp/sccache\
ENV SCCACHE_CACHE_SIZE=10G' \
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

echo "Building ${IMAGE_NAME} Docker image (local build)"

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
