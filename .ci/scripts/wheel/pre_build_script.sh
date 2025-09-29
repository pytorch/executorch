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

# ----------------------------
# Stage prebuilt glibc 2.34
# ----------------------------
set -euo pipefail

GLIBC_VERSION=2.34
PREFIX=/tmp/glibc-install-$GLIBC_VERSION
mkdir -p "$PREFIX/lib"

echo ">>> Downloading prebuilt glibc-$GLIBC_VERSION (Fedora 35 RPM)"
RPM_URL="https://archives.fedoraproject.org/pub/archive/fedora/linux/releases/35/Everything/x86_64/os/Packages/g/glibc-2.34-7.fc35.x86_64.rpm"

# Download
curl -fsSL "$RPM_URL" -o /tmp/glibc.rpm

# Extract directly with bsdtar
echo ">>> Extracting RPM with bsdtar"
bsdtar -C /tmp -xf /tmp/glibc.rpm

# Copy needed files from the extracted tree (not host system!)
# Copy all runtime libs from extracted RPM
cp -av /tmp/lib64/libc.so.6 \
       /tmp/lib64/ld-linux-x86-64.so.2 \
       /tmp/lib64/libdl.so.2 \
       /tmp/lib64/libpthread.so.0 \
       /tmp/lib64/librt.so.1 \
       /tmp/lib64/libm.so.6 \
       /tmp/lib64/libutil.so.1 \
       "$PREFIX/lib/" || true

# Check what we staged
echo ">>> Contents staged in $PREFIX/lib"
ls -l "$PREFIX/lib"

# Verify version
"$PREFIX/lib/libc.so.6" --version || true
