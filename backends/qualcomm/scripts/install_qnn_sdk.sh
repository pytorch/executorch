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

    # dl.google.com intermittently resets HTTP/2 streams mid-transfer
    # (curl error 92, INTERNAL_ERROR). Force HTTP/1.1 to avoid it, and use
    # --retry-all-errors so the retry count actually applies to such transport
    # failures (plain --retry does not retry them). Re-download and re-verify
    # the archive on each attempt rather than resuming a possibly-corrupt file.
    for attempt in 1 2 3 4 5; do
        rm -f "/tmp/${NDK_ZIP}"
        curl --fail --http1.1 --retry 3 --retry-delay 5 --retry-connrefused --retry-all-errors \
            -Lo "/tmp/${NDK_ZIP}" "https://dl.google.com/android/repository/${NDK_ZIP}" || true
        if unzip -tq "/tmp/${NDK_ZIP}" >/dev/null 2>&1; then
            break
        fi
        if [ "${attempt}" -eq 5 ]; then
            echo "Failed to download a valid Android NDK archive after ${attempt} attempts" >&2
            exit 1
        fi
        echo "NDK download/verify failed (attempt ${attempt}), retrying..."
        sleep 5
    done
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
  # softwarecenter.qualcomm.com intermittently aborts the download with
  # HTTP/2 INTERNAL_ERROR mid-stream, and occasionally returns a tiny
  # error body that curl treats as success — both cases get caught here:
  # --fail rejects HTTP errors, --retry-all-errors retries transport
  # errors, and `unzip -t` validates the archive before we proceed.
  QNN_DOWNLOAD_MAX_ATTEMPTS=5
  for attempt in $(seq 1 ${QNN_DOWNLOAD_MAX_ATTEMPTS}); do
    rm -f "/tmp/${QNN_ZIP_FILE}"
    if curl --fail --retry 3 --retry-delay 5 --retry-connrefused --retry-all-errors \
         -Lo "/tmp/${QNN_ZIP_FILE}" "${QNN_ZIP_URL}" \
       && unzip -tq "/tmp/${QNN_ZIP_FILE}"; then
      break
    fi
    ls -l "/tmp/${QNN_ZIP_FILE}" 2>&1 || true
    if [ "${attempt}" = "${QNN_DOWNLOAD_MAX_ATTEMPTS}" ]; then
      echo "ERROR: QNN SDK download failed after ${attempt} attempts" >&2
      exit 1
    fi
    echo "QNN SDK download attempt ${attempt} failed; retrying in $((attempt * 10))s..."
    sleep $((attempt * 10))
  done
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

# Install the Hexagon SDK required for direct-mode CI builds.
install_hexagon_sdk() {
  # Check if already configured externally and valid.
  if [ -n "${HEXAGON_SDK_ROOT:-}" ] && [ -d "${HEXAGON_SDK_ROOT:-}" ] \
     && [ -n "${HEXAGON_TOOLS_ROOT:-}" ] && [ -d "${HEXAGON_TOOLS_ROOT:-}" ]; then
    echo "Hexagon SDK already set to ${HEXAGON_SDK_ROOT} - skipping installation"
    return
  fi

  echo "Start installing Hexagon SDK v${HEXAGON_SDK_VERSION} (tools v${HEXAGON_TOOLS_VERSION})"
  HEXAGON_INSTALLATION_DIR="/tmp/hexagon-sdk"
  HEXAGON_SDK_DIR="${HEXAGON_INSTALLATION_DIR}/Hexagon_SDK/${HEXAGON_SDK_VERSION}"
  HEXAGON_TOOLS_DIR="${HEXAGON_SDK_DIR}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}"

  # Return if already exist
  if [ -d "${HEXAGON_SDK_DIR}" ] && [ -d "${HEXAGON_TOOLS_DIR}" ]; then
    echo "Hexagon SDK already installed at ${HEXAGON_SDK_DIR}"
    export HEXAGON_SDK_ROOT="${HEXAGON_SDK_DIR}"
    export HEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_DIR}"
    return
  fi

  mkdir -p "${HEXAGON_INSTALLATION_DIR}"

  HEXAGON_ZIP_FILE="Hexagon_SDK_Linux.zip"
  # Match install_qnn's retry shape: --fail rejects HTTP errors,
  # --retry-all-errors retries transport failures, `unzip -t` validates the
  # archive, and the SHA-256 check pins the exact bytes we tested against. All
  # are inside the retry condition so a truncated or wrong-content download is
  # re-fetched rather than killing the job.
  HEXAGON_DOWNLOAD_MAX_ATTEMPTS=5
  for attempt in $(seq 1 ${HEXAGON_DOWNLOAD_MAX_ATTEMPTS}); do
    rm -f "/tmp/${HEXAGON_ZIP_FILE}"
    if curl --fail --retry 3 --retry-delay 5 --retry-connrefused --retry-all-errors \
         -Lo "/tmp/${HEXAGON_ZIP_FILE}" "${HEXAGON_SDK_ZIP_URL}" \
       && unzip -tq "/tmp/${HEXAGON_ZIP_FILE}" \
       && echo "${HEXAGON_SDK_ZIP_SHA256}  /tmp/${HEXAGON_ZIP_FILE}" | sha256sum -c -; then
      break
    fi
    ls -l "/tmp/${HEXAGON_ZIP_FILE}" 2>&1 || true
    if [ "${attempt}" = "${HEXAGON_DOWNLOAD_MAX_ATTEMPTS}" ]; then
      echo "ERROR: Hexagon SDK download failed after ${attempt} attempts" >&2
      exit 1
    fi
    echo "Hexagon SDK download attempt ${attempt} failed; retrying in $((attempt * 10))s..."
    sleep $((attempt * 10))
  done
  echo "Finishing downloading Hexagon SDK."

  unzip -qo "/tmp/${HEXAGON_ZIP_FILE}" -d "${HEXAGON_INSTALLATION_DIR}"
  echo "Finishing unzip Hexagon SDK."

  export HEXAGON_SDK_ROOT="${HEXAGON_SDK_DIR}"
  export HEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_DIR}"

  # Verify the unzipped layout matches what build.sh and the QNN CMake
  # files actually consume. If any of these are missing, a future SDK
  # release likely changed the directory shape; updating
  # HEXAGON_SDK_VERSION / HEXAGON_TOOLS_VERSION in qnn_config.sh (or the
  # extraction layout below) is the fix.
  for hexagon_required_path in \
      "${HEXAGON_SDK_ROOT}" \
      "${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake" \
      "${HEXAGON_TOOLS_ROOT}" \
      "${HEXAGON_TOOLS_ROOT}/Tools/target/hexagon"; do
    if [ ! -e "${hexagon_required_path}" ]; then
      echo "[Hexagon] ERROR: expected path not found: ${hexagon_required_path}" >&2
      echo "[Hexagon] Hexagon SDK ${HEXAGON_SDK_VERSION} or tools ${HEXAGON_TOOLS_VERSION} layout differs from what we pinned." >&2
      ls -la "$(dirname "${hexagon_required_path}")" >&2 || true
      exit 1
    fi
  done

  echo "Set HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}"
  echo "Set HEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT}"
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
  curl --retry 3 -fLO "${LLVM_URL}" || {
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
