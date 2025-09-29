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

echo ">>> Downloading prebuilt glibc-$GLIBC_VERSION (CentOS Stream 9)"

# Choose a mirror (can switch if one dies)
RPM_URL="http://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os/Packages/glibc-${GLIBC_VERSION}-100.el9.x86_64.rpm"

# Download RPM
curl -L "$RPM_URL" -o /tmp/glibc.rpm

# Extract without root
command -v rpm2cpio >/dev/null 2>&1 || {
    echo ">>> rpm2cpio not found, downloading static helper..."
    curl -L https://raw.githubusercontent.com/rpm-software-management/rpm/main/scripts/rpm2cpio.sh -o /tmp/rpm2cpio.sh
    chmod +x /tmp/rpm2cpio.sh
    RPM2CPIO=/tmp/rpm2cpio.sh
}
RPM2CPIO=${RPM2CPIO:-rpm2cpio}

$RPM2CPIO /tmp/glibc.rpm | cpio -idmv

# Copy only runtime loader + libc
cp ./usr/lib64/libc.so.6 \
   ./usr/lib64/ld-2.34.so \
   ./usr/lib64/ld-linux-x86-64.so.2 \
   "$PREFIX/lib"

echo ">>> Staged glibc $GLIBC_VERSION to $PREFIX/lib"
ls -l "$PREFIX/lib"

# Verify
"$PREFIX/lib/libc.so.6" --version || true
"$PREFIX/lib/ld-2.34.so" --version || true
