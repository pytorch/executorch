#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

# This script is run before building ExecuTorch binaries

# Initialize submodules here instead of during checkout so we can use OpenSSL
# on Windows (schannel fails with SEC_E_ILLEGAL_MESSAGE on some gitlab hosts).
UNAME_S=$(uname -s)
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
  git -c http.sslBackend=openssl submodule update --init
else
  git submodule update --init
fi

# Clone nested submodules for tokenizers - this is a workaround for recursive
# submodule clone failing due to path length limitations on Windows. Eventually,
# we should update the core job in test-infra to enable long paths before
# checkout to avoid needing to do this.
pushd extension/llm/tokenizers
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
  git -c http.sslBackend=openssl submodule update --init
else
  git submodule update --init
fi
popd

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

# On Windows, enable symlinks and re-checkout the current revision to create
# the symlinked src/ directory. This is needed to build the wheel.
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

# Enable VGF in pybind wheel builds when the platform-specific build input is
# available from pip.
if [[ "$UNAME_S" == "Linux" || "$UNAME_S" == "Darwin" ]]; then
  if python3 -m pip install -r \
    "${GITHUB_WORKSPACE}/${REPOSITORY}/backends/arm/requirements-arm-vgf-runtime.txt"; then
    export EXECUTORCH_PYBIND_ENABLE_VGF=ON
    echo "EXECUTORCH_PYBIND_ENABLE_VGF=ON" >> "${GITHUB_ENV}"
  else
    echo "VGF build dependency unavailable on this platform; building without VGF"
  fi
fi

# Download Qualcomm QNN SDK on Linux x86_64 so the wheel build can include the
# QNN backend.  The SDK is large, so we download it here (outside CMake) rather
# than during cmake configure.
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  echo "Downloading Qualcomm QNN SDK..."
  QNN_SDK_ROOT=$(python3 \
    "${GITHUB_WORKSPACE}/${REPOSITORY}/backends/qualcomm/scripts/download_qnn_sdk.py" \
    --print-sdk-path)
  export QNN_SDK_ROOT
  echo "QNN_SDK_ROOT=${QNN_SDK_ROOT}" >> "${GITHUB_ENV}"
  echo "QNN SDK downloaded to ${QNN_SDK_ROOT}"
fi
