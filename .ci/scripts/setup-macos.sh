#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

read -r BUILD_TOOL BUILD_MODE EDITABLE < <(parse_args "$@")

install_buck() {
  if ! command -v zstd &> /dev/null; then
    brew install zstd
  fi

  if ! command -v curl &> /dev/null; then
    brew install curl
  fi

  python -m pip install certifi

  pushd .ci/docker

  # Use resolve_buck.py to download and install buck2
  # The script handles platform detection and version management
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

  # Create a cache directory for buck2
  CACHE_DIR="${HOME}/.cache/buck2"
  mkdir -p "${CACHE_DIR}"

  # Run resolve_buck.py to get the buck2 binary
  BUCK2_PATH=$(python "${REPO_ROOT}/tools/cmake/resolve_buck.py" --cache_dir "${CACHE_DIR}")

  # Move buck2 to /opt/homebrew/bin
  sudo mv "${BUCK2_PATH}" /opt/homebrew/bin/buck2
  sudo chmod +x /opt/homebrew/bin/buck2

  popd

  # Kill all running buck2 daemon for a fresh start
  buck2 killall || true
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
brew install libomp
install_pip_dependencies

# TODO(huydhn): Unlike our self-hosted runner, GitHub runner doesn't have access
# to our infra, so compiler caching needs to be setup differently using GitHub
# cache. However, I need to figure out how to set that up for Nova MacOS job
if [[ -z "${GITHUB_RUNNER:-}" ]]; then
  install_sccache
fi

print_cmake_info
install_pytorch_and_domains
# We build PyTorch from source here instead of using nightly. This allows CI to test against
# the pinned commit from PyTorch
if [[ "$EDITABLE" == "true" ]]; then
  install_executorch --use-pt-pinned-commit --editable
else
  install_executorch --use-pt-pinned-commit
fi
build_executorch_runner "${BUILD_TOOL}" "${BUILD_MODE}"

if [[ "${GITHUB_BASE_REF:-}" == *main* || "${GITHUB_BASE_REF:-}" == *gh* ]]; then
  do_not_use_nightly_on_ci
fi
