#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

# This script is run before building ExecuTorch binaries

if [[ "$(uname -m)" == "aarch64" ]]; then
  # On some Linux aarch64 systems, the "atomic" library is not found during linking.
  # To work around this, replace "atomic" with the literal ${ATOMIC_LIB} so the
  # build system uses the full path to the atomic library.
  file="extension/llm/tokenizers/third-party/sentencepiece/src/CMakeLists.txt"
  sed 's/list(APPEND SPM_LIBS "atomic")/list(APPEND SPM_LIBS ${ATOMIC_LIB})/' \
    "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"

  grep -n 'list(APPEND SPM_LIBS ${ATOMIC_LIB})' "$file" && \
    echo "the file $file has been modified for atomic to use full path"
fi

# Clone nested submodules for tokenizers - this is a workaround for recursive
# submodule clone failing due to path length limitations on Windows. Eventually,
# we should update the core job in test-infra to enable long paths before
# checkout to avoid needing to do this.
pushd extension/llm/tokenizers
git submodule update --init
popd

# On Windows, enable symlinks and re-checkout the current revision to create
# the symlinked src/ directory. This is needed to build the wheel.
UNAME_S=$(uname -s)
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
    echo "Enabling symlinks on Windows"
    git config core.symlinks true
    git checkout -f HEAD
fi

# Manually install build requirements because `python setup.py bdist_wheel` does
# not install them. TODO(dbort): Switch to using `python -m build --wheel`,
# which does install them. Though we'd need to disable build isolation to be
# able to see the installed torch package.

"${GITHUB_WORKSPACE}/${REPOSITORY}/install_requirements.sh" --example

#!/usr/bin/env bash
set -euxo pipefail

GLIBC_VERSION=2.29
PREFIX=/tmp/glibc-$GLIBC_VERSION
BUILD_DIR=/tmp/glibc-build

# Clean old dirs if they exist
rm -rf "$PREFIX" "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Download
curl -sSL https://ftp.gnu.org/gnu/libc/glibc-$GLIBC_VERSION.tar.xz -o /tmp/glibc-$GLIBC_VERSION.tar.xz
tar -C /tmp -xf /tmp/glibc-$GLIBC_VERSION.tar.xz

cd "$BUILD_DIR"

# Configure with relaxed flags
CFLAGS="-O2 -fPIC -fcommon -Wno-error=array-parameter -Wno-error=stringop-overflow -Wno-error" \
../glibc-$GLIBC_VERSION/configure --prefix="$PREFIX"

# Build
make -j"$(nproc)" CFLAGS="$CFLAGS"

# Install
make install

# Check
"$PREFIX/lib/ld-2.29.so" --version || true
"$PREFIX/lib/libc.so.6" --version || true
