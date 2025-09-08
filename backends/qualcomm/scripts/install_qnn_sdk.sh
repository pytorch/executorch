set -ex

# Get the absolute path of this script
SCRIPT_DIR="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"

# Source QNN configuration from the same directory
source "${SCRIPT_DIR}/qnn_config.sh"

# Function to install Android NDK (only if not already set)
setup_android_ndk() {
    # Check if ANDROID_NDK_ROOT is already set and valid
    if [ -n "${ANDROID_NDK_ROOT:-}" ] && [ -d "${ANDROID_NDK_ROOT:-}" ]; then
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
  if [ -n "${QNN_SDK_ROOT:-}" ] && [ -d "${QNN_SDK_ROOT:-}" ]; then
    echo "QNN SDK already set to ${QNN_SDK_ROOT} - skipping installation"
    return
  fi

  echo "Start installing qnn v${QNN_VERSION}"
  QNN_INSTALLATION_DIR="/tmp/qnn"
  
  if [ -d "${QNN_INSTALLATION_DIR}/${QNN_VERSION}" ]; then
        echo "QNN SDK already installed at ${QNN_INSTALLATION_DIR}/${QNN_VERSION}"
        export QNN_SDK_ROOT="${QNN_INSTALLATION_DIR}/${QNN_VERSION}"
        return
  fi

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
  export QNN_SDK_ROOT="${QNN_INSTALLATION_DIR}/${QNN_VERSION}"
  echo "Set QNN_SDK_ROOT=${QNN_SDK_ROOT}"
}

setup_libcpp() {
  clang_version=$1
  LLVM_VERSION="14.0.0"
  INSTALL_DIR="/tmp/libcxx-${LLVM_VERSION}"

  # Check if we already have a local installation
  if [ -d "${INSTALL_DIR}/include" ] && [ -d "${INSTALL_DIR}/lib" ]; then
    echo "Local libc++ already installed at ${INSTALL_DIR} - skipping"
    # Set environment variables
    export CPLUS_INCLUDE_PATH="${INSTALL_DIR}/include:$CPLUS_INCLUDE_PATH"
    export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="${INSTALL_DIR}/lib:$LIBRARY_PATH"
    return
  fi

  echo "Installing libc++ manually to ${INSTALL_DIR}"

  # Create temporary directory
  TEMP_DIR=$(mktemp -d)
  # Ensure cleanup on exit or return
  trap 'rm -rf "$TEMP_DIR"' RETURN

  pushd "${TEMP_DIR}" >/dev/null

  BASE_NAME="clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04"
  LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${BASE_NAME}.tar.xz"

  echo "Downloading LLVM from ${LLVM_URL}"
  curl -fLO "${LLVM_URL}" || {
      echo "Error: Failed to download LLVM"
      exit 1
  }

  echo "Extracting ${BASE_NAME}.tar.xz"
  tar -xf "${BASE_NAME}.tar.xz" || {
      echo "Error: Failed to extract LLVM archive"
      exit 1
  }

  # Create installation directory
  mkdir -p "${INSTALL_DIR}/include"
  mkdir -p "${INSTALL_DIR}/lib"

  # Copy libc++ headers and libraries
  cp -r "${BASE_NAME}/include/c++/v1/"* "${INSTALL_DIR}/include/"
  cp -r "${BASE_NAME}/lib/"*.so* "${INSTALL_DIR}/lib/"

  popd >/dev/null

  # Create necessary symlinks locally
  pushd "${INSTALL_DIR}/lib" >/dev/null
  ln -sf libc++.so.1.0 libc++.so.1
  ln -sf libc++.so.1 libc++.so
  ln -sf libc++abi.so.1.0 libc++abi.so.1
  ln -sf libc++abi.so.1 libc++abi.so
  popd >/dev/null

  # Set environment variables
  export CPLUS_INCLUDE_PATH="${INSTALL_DIR}/include:${CPLUS_INCLUDE_PATH:-}"
  export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${INSTALL_DIR}/lib:${LIBRARY_PATH:-}"

  echo "libc++ installed to ${INSTALL_DIR}"
}