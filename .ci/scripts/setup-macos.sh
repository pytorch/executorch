#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

BUILD_TOOL=$1
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
else
  echo "Setup MacOS for ${BUILD_TOOL} ..."
fi

install_buck() {
  if ! command -v zstd &> /dev/null; then
    brew install zstd
  fi

  if ! command -v curl &> /dev/null; then
    brew install curl
  fi

  pushd .ci/docker
  # TODO(huydo): This is a one-off copy of buck2 2024-05-15 to unblock Jon and
  # re-enable ShipIt. It’s not ideal that upgrading buck2 will require a manual
  # update the cached binary on S3 bucket too. Let me figure out if there is a
  # way to correctly implement the previous setup of installing a new version of
  # buck2 only when it’s needed. AFAIK, the complicated part was that buck2
  # --version doesn't say anything w.r.t its release version, i.e. 2024-05-15.
  # See D53878006 for more details.
  #
  # If you need to upgrade buck2 version on S3, please reach out to Dev Infra
  # team for help.
  BUCK2_VERSION=$(cat ci_commit_pins/buck2.txt)
  BUCK2=buck2-aarch64-apple-darwin-${BUCK2_VERSION}.zst
  curl -s "https://ossci-macos.s3.amazonaws.com/${BUCK2}" -o "${BUCK2}"

  zstd -d "${BUCK2}" -o buck2

  chmod +x buck2
  mv buck2 /opt/homebrew/bin

  rm "${BUCK2}"
  popd
}

function write_sccache_stub() {
  OUTPUT=$1
  BINARY=$(basename "${OUTPUT}")

  printf "#!/bin/sh\nif [ \$(ps auxc \$(ps auxc -o ppid \$\$ | grep \$\$ | rev | cut -d' ' -f1 | rev) | tr '\\\\n' ' ' | rev | cut -d' ' -f2 | rev) != sccache ]; then\n  exec sccache %s \"\$@\"\nelse\n  exec %s \"\$@\"\nfi" "$(which "${BINARY}")" "$(which "${BINARY}")" > "${OUTPUT}"
  chmod a+x "${OUTPUT}"
}

install_sccache() {
  # Use existing S3 cache bucket for self-hosted MacOS runner
  export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
  export SCCACHE_S3_KEY_PREFIX=executorch
  export SCCACHE_IDLE_TIMEOUT=0
  export SCCACHE_ERROR_LOG=/tmp/sccache_error.log
  export RUST_LOG=sccache::server=error

  SCCACHE_PATH="/usr/local/bin"
  # NB: The function is adopted from PyTorch MacOS build workflow
  # https://github.com/pytorch/pytorch/blob/main/.github/workflows/_mac-build.yml
  if ! command -v sccache &> /dev/null; then
    sudo curl --retry 3 "https://s3.amazonaws.com/ossci-macos/sccache/sccache-v0.4.1-${RUNNER_ARCH}" --output "${SCCACHE_PATH}/sccache"
    sudo chmod +x "${SCCACHE_PATH}/sccache"
  fi

  export PATH="${SCCACHE_PATH}:${PATH}"

  # Create temp directory for sccache shims if TMP_DIR doesn't exist
  if [ -z "${TMP_DIR:-}" ]; then
    TMP_DIR=$(mktemp -d)
    trap 'rm -rfv ${TMP_DIR}' EXIT
    export PATH="${TMP_DIR}:$PATH"
  fi

  write_sccache_stub "${TMP_DIR}/c++"
  write_sccache_stub "${TMP_DIR}/cc"
  write_sccache_stub "${TMP_DIR}/clang++"
  write_sccache_stub "${TMP_DIR}/clang"

  sccache --zero-stats || true
}

# This is the same rpath fix copied from PyTorch macos setup script
# https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/macos-common.sh
print_cmake_info() {
  CMAKE_EXEC=$(which cmake)
  echo "$CMAKE_EXEC"

  export CMAKE_EXEC
  # Explicitly add conda env lib folder to cmake rpath to address the flaky issue
  # where cmake dependencies couldn't be found. This seems to point to how conda
  # links $CMAKE_EXEC to its package cache when cloning a new environment
  install_name_tool -add_rpath @executable_path/../lib "${CMAKE_EXEC}" || true
  # Adding the rpath will invalidate cmake signature, so signing it again here
  # to trust the executable. EXC_BAD_ACCESS (SIGKILL (Code Signature Invalid))
  # with an exit code 137 otherwise
  codesign -f -s - "${CMAKE_EXEC}" || true
}

setup_macos_env_variables() {
  CMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')
  export CMAKE_PREFIX_PATH
}

setup_macos_env_variables
# NB: we need buck2 in all cases because cmake build also depends on calling
# buck2 atm
install_buck
install_pip_dependencies

# TODO(huydhn): Unlike our self-hosted runner, GitHub runner doesn't have access
# to our infra, so compiler caching needs to be setup differently using GitHub
# cache. However, I need to figure out how to set that up for Nova MacOS job
if [[ -z "${GITHUB_RUNNER:-}" ]]; then
  install_sccache
fi

print_cmake_info
install_executorch
build_executorch_runner "${BUILD_TOOL}"
