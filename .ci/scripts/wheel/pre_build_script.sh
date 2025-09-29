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

# Install glibc-2.34

# Install older GCC (works on GitHub Ubuntu runners)
apt-get update
apt-get install -y gcc-11 g++-11

# Force glibc build to use GCC 11
export CC=gcc-11
export CXX=g++-11

echo "GCC version (system default): $(gcc -dumpfullversion)"
echo "GCC version (forced CC): $($CC -dumpfullversion)"

# 👇 only change this line to bump version
GLIBC_VERSION=2.34

PREFIX=/tmp/glibc-install-$GLIBC_VERSION
BUILD_DIR=/tmp/glibc-build
TARBALL=/tmp/glibc-$GLIBC_VERSION.tar.xz
SRC_DIR=/tmp/glibc-$GLIBC_VERSION

# Clean old dirs
rm -rf "$PREFIX" "$BUILD_DIR" "$SRC_DIR" "$TARBALL"
mkdir -p "$BUILD_DIR"

# Download tarball from canonical GNU FTP
MIRROR=https://ftp.gnu.org/gnu/libc
curl -L "$MIRROR/glibc-$GLIBC_VERSION.tar.xz" -o "$TARBALL"

# Sanity check tarball
ls -lh "$TARBALL"
file "$TARBALL" || true

# Extract
tar -C /tmp -xf "$TARBALL"

cd "$BUILD_DIR"

# Unset LD_LIBRARY_PATH to satisfy glibc configure
unset LD_LIBRARY_PATH

# Suppress GCC warnings (GCC 11 is more compatible, but keep flags)
COMMON_FLAGS="-O2 -fPIC -fcommon -Wno-error"
export CFLAGS="$COMMON_FLAGS"
export CPPFLAGS="$COMMON_FLAGS"
export CXXFLAGS="$COMMON_FLAGS"

# Configure
../glibc-$GLIBC_VERSION/configure \
    --prefix="$PREFIX" \
    --without-selinux

# Build and install
make -j"$(nproc)"
make install

echo ">>> Finished make install"
echo ">>> PREFIX=$PREFIX"

# List everything under $PREFIX
echo ">>> ls -l $PREFIX"
ls -l "$PREFIX" || true

echo ">>> ls -l $PREFIX/lib"
ls -l "$PREFIX/lib" || true

echo ">>> ls -l $PREFIX/lib64"
ls -l "$PREFIX/lib64" || true

# Explicitly show libc/ld files
echo ">>> Checking for libc.so.6"
if [ -e "$PREFIX/lib/libc.so.6" ]; then
    ls -l "$PREFIX/lib/libc.so.6"
else
    echo "libc.so.6 NOT FOUND in $PREFIX/lib"
fi

echo ">>> Checking for ld-$GLIBC_VERSION.so"
if [ -e "$PREFIX/lib/ld-$GLIBC_VERSION.so" ]; then
    ls -l "$PREFIX/lib/ld-$GLIBC_VERSION.so"
else
    echo "ld-$GLIBC_VERSION.so NOT FOUND in $PREFIX/lib"
fi

# Run version checks only if files exist
[ -x "$PREFIX/lib/ld-$GLIBC_VERSION.so" ] && "$PREFIX/lib/ld-$GLIBC_VERSION.so" --version || echo "ld-$GLIBC_VERSION.so missing or not executable"
[ -x "$PREFIX/lib/libc.so.6" ] && "$PREFIX/lib/libc.so.6" --version || echo "libc.so.6 missing or not executable"
