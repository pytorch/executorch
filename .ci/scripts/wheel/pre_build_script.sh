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

# Print GCC version (just the number)
echo "GCC version: $(gcc -dumpfullversion)"

# Version / directories
GLIBC_VERSION=2.29
PREFIX=/tmp/glibc-$GLIBC_VERSION
BUILD_DIR=/tmp/glibc-build
TARBALL=/tmp/glibc-$GLIBC_VERSION.tar.xz
SRC_DIR=/tmp/glibc-$GLIBC_VERSION

# Clean old dirs/files if they exist
rm -rf "$PREFIX" "$BUILD_DIR" "$SRC_DIR" "$TARBALL"
mkdir -p "$BUILD_DIR"

# Download source tarball into /tmp
MIRROR=https://ftpmirror.gnu.org/gnu/libc
curl -L "$MIRROR/glibc-$GLIBC_VERSION.tar.xz" -o "$TARBALL"

# Extract into /tmp
tar -C /tmp -xf "$TARBALL"

# Configure with relaxed flags
cd "$BUILD_DIR"
CFLAGS="-O2 -fPIC -fcommon -Wno-error=array-parameter -Wno-error=stringop-overflow -Wno-error" \
  ../glibc-$GLIBC_VERSION/configure --prefix="$PREFIX"

# Build & install
make -j"$(nproc)" CFLAGS="$CFLAGS"
make install

# Quick check
"$PREFIX/lib/ld-2.29.so" --version || true
"$PREFIX/lib/libc.so.6" --version || true
