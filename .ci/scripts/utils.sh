#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

reset_buck() {
  # On MacOS, buck2 daemon can get into a weird non-responsive state
  buck2 kill && buck2 clean
  rm -rf ~/.buck/buckd
}

retry () {
    "$@" || (sleep 30 && reset_buck && "$@") || (sleep 60 && reset_buck && "$@")
}

install_executorch() {
  which pip
  # Install executorch, this assumes that Executorch is checked out in the
  # current directory
  pip install .
  # Just print out the list of packages for debugging
  pip list
}

install_conda() {
  pushd .ci/docker || return
  # Install conda dependencies like flatbuffer
  conda install -y --file conda-env-ci.txt
  popd || return
}

install_pip_dependencies() {
  pushd .ci/docker || return
  # Install all Python dependencies, including PyTorch
  pip install --progress-bar off -r requirements-ci.txt

  NIGHTLY=$(cat ci_commit_pins/nightly.txt)
  TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
  TORCHAUDIO_VERSION=$(cat ci_commit_pins/audio.txt)
  TORCHVISION_VERSION=$(cat ci_commit_pins/vision.txt)

  pip install --progress-bar off --pre \
    torch=="${TORCH_VERSION}.${NIGHTLY}" \
    torchaudio=="${TORCHAUDIO_VERSION}.${NIGHTLY}" \
    torchvision=="${TORCHVISION_VERSION}.${NIGHTLY}" \
    --index-url https://download.pytorch.org/whl/nightly/cpu
  popd || return
}

install_flatc_from_source() {
  # NB: This function could be used to install flatbuffer from source
  pushd third-party/flatbuffers || return

  cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build . -j "${CMAKE_JOBS}"

  # Copy the flatc binary to conda path
  EXEC_PATH=$(dirname "$(which python)")
  cp flatc "${EXEC_PATH}"

  popd || return
}

build_executorch_runner_buck2() {
  # Build executorch runtime with retry as this step is flaky on macos CI
  retry buck2 build //examples/portable/executor_runner:executor_runner
}

build_executorch_runner_cmake() {
  CMAKE_OUTPUT_DIR=cmake-out
  # Build executorch runtime using cmake
  rm -rf "${CMAKE_OUTPUT_DIR}" && mkdir "${CMAKE_OUTPUT_DIR}"

  pushd "${CMAKE_OUTPUT_DIR}" || return
  # This command uses buck2 to gather source files and buck2 could crash flakily
  # on MacOS
  retry cmake -DBUCK2=buck2 -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -DCMAKE_BUILD_TYPE=Release ..
  popd || return

  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build "${CMAKE_OUTPUT_DIR}" -j "${CMAKE_JOBS}"
}

build_executorch_runner() {
  if [[ $1 == "buck2" ]]; then
    build_executorch_runner_buck2
  elif [[ $1 == "cmake" ]]; then
    build_executorch_runner_cmake
  else
    echo "Invalid build tool $1. Only buck2 and cmake are supported atm"
    exit 1
  fi
}
