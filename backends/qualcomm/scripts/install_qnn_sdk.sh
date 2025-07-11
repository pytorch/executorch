#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Get the absolute path of this script
SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"

# Source QNN configuration from the same directory
source "${SCRIPT_DIR}/qnn_config.sh"

# Function to install Android NDK (only if not already set)
setup_android_ndk() {
    # Check if ANDROID_NDK_ROOT is already set and valid
    if [ -n "${ANDROID_NDK_ROOT}" ] && [ -d "${ANDROID_NDK_ROOT}" ]; then
        echo "Android NDK already set to ${ANDROID_NDK_ROOT} - skipping installation"
        return
    fi

    NDK_VERSION="r26c"
    NDK_INSTALL_DIR="/tmp/android-ndk"

    if [ -d "${NDK_INSTALL_DIR}/ndk" ]; then
        echo "Android NDK already installed at ${NDK_INSTALL_DIR}/ndk"
        export ANDROID_NDK_ROOT="${NDK_INSTALL_DIR}/ndk"
        return
    fi

    echo "Installing Android NDK ${NDK_VERSION}"
    mkdir -p "${NDK_INSTALL_DIR}"
    NDK_ZIP="android-ndk-${NDK_VERSION}-linux.zip"

    curl -Lo "/tmp/${NDK_ZIP}" "https://dl.google.com/android/repository/${NDK_ZIP}"
    unzip -q "/tmp/${NDK_ZIP}" -d "${NDK_INSTALL_DIR}"
    mv "${NDK_INSTALL_DIR}/android-ndk-${NDK_VERSION}" "${NDK_INSTALL_DIR}/ndk"

    export ANDROID_NDK_ROOT="${NDK_INSTALL_DIR}/ndk"
    echo "Android NDK installed to ${ANDROID_NDK_ROOT}"
}

verify_pkg_installed() {
  dpkg-query -W --showformat='${Status}\n' "$1" | grep -q "install ok installed"
}

install_qnn() {
  # Check if QNN_SDK_ROOT is already set and valid
  if [ -n "${QNN_SDK_ROOT}" ] && [ -d "${QNN_SDK_ROOT}" ]; then
    echo "QNN SDK already set to ${QNN_SDK_ROOT} - skipping installation"
    return
  fi

  echo "Start installing qnn v${QNN_VERSION}"
  QNN_INSTALLATION_DIR="/tmp/qnn"

  # Clean up any previous installation
  if [ -d "${QNN_INSTALLATION_DIR}" ]; then
    echo "Removing previous QNN installation at ${QNN_INSTALLATION_DIR}"
    rm -rf "${QNN_INSTALLATION_DIR}"
  fi

  mkdir -p "${QNN_INSTALLATION_DIR}"

  QNN_ZIP_FILE="v${QNN_VERSION}.zip"
  curl -Lo "/tmp/${QNN_ZIP_FILE}" "${QNN_ZIP_URL}"
  echo "Finishing downloading qnn sdk."
  unzip -qo "/tmp/${QNN_ZIP_FILE}" -d /tmp
  echo "Finishing unzip qnn sdk."

  # Print the content for manual verification
  echo "Contents of /tmp/qairt:"
  ls -lah "/tmp/qairt"

  # Move the specific version directory
  if [ -d "/tmp/qairt/${QNN_VERSION}" ]; then
    mv "/tmp/qairt/${QNN_VERSION}" "${QNN_INSTALLATION_DIR}"
  else
    mv "/tmp/qairt"/* "${QNN_INSTALLATION_DIR}"
  fi

  echo "Finishing installing qnn '${QNN_INSTALLATION_DIR}' ."
  echo "Final QNN installation contents:"
  ls -lah "${QNN_INSTALLATION_DIR}"

  # Set QNN_SDK_ROOT environment variable
  export QNN_SDK_ROOT="${QNN_INSTALLATION_DIR}"
  echo "Set QNN_SDK_ROOT=${QNN_SDK_ROOT}"
}

setup_libcpp() {
  clang_version=$1
  LLVM_VERSION="14.0.0"
  INSTALL_DIR="/usr/local/libcxx-${LLVM_VERSION}"

  # Check if libc++ is already installed
  if [ -d "/usr/include/c++/v1" ] && \
     [ -f "/usr/lib/libc++.so.1" ] && \
     [ -f "/usr/lib/libc++abi.so.1" ]; then
    echo "libc++-${clang_version}-dev is already installed - skipping"
    return
  fi

  echo "Installing libc++-${clang_version}-dev manually from LLVM releases"

  # Create temporary directory
  TEMP_DIR=$(mktemp -d)
  pushd "${TEMP_DIR}"

  # Download and extract LLVM binaries
  LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-20.04.tar.xz"
  curl -LO "${LLVM_URL}"
  tar -xf "clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-20.04.tar.xz"

  # Create necessary directories
  sudo mkdir -p "${INSTALL_DIR}/include"
  sudo mkdir -p "${INSTALL_DIR}/lib"  # FIX: Create lib directory

  # Copy libc++ headers and libraries
  sudo cp -r clang+llvm*/include/c++/v1/* "${INSTALL_DIR}/include/"
  sudo cp -r clang+llvm*/lib/*.so* "${INSTALL_DIR}/lib/"

  # Create system symlinks
  sudo mkdir -p /usr/include/c++
  sudo ln -sf "${INSTALL_DIR}/include" /usr/include/c++/v1
  sudo ln -sf "${INSTALL_DIR}/lib/libc++.so.1.0" /usr/lib/libc++.so.1
  sudo ln -sf "${INSTALL_DIR}/lib/libc++.so.1" /usr/lib/libc++.so
  sudo ln -sf "${INSTALL_DIR}/lib/libc++abi.so.1.0" /usr/lib/libc++abi.so.1
  sudo ln -sf "${INSTALL_DIR}/lib/libc++abi.so.1" /usr/lib/libc++abi.so

  # Update library cache
  sudo ldconfig

  # Cleanup
  popd
  rm -rf "${TEMP_DIR}"

  echo "libc++-${clang_version}-dev installed to ${INSTALL_DIR}"
}

setup_libcpp 12
setup_android_ndk
install_qnn
