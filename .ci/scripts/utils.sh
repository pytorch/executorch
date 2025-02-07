#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024 Arm Limited and/or its affiliates.
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

clean_executorch_install_folders() {
  ./install_executorch.sh --clean
}

install_executorch() {
  which pip
  # Install executorch, this assumes that Executorch is checked out in the
  # current directory.
  if [[ "${1:-}" == "use-pt-pinned-commit" ]]; then
    ./install_executorch.sh --pybind xnnpack --use-pt-pinned-commit
  else
    ./install_executorch.sh --pybind xnnpack
  fi
  # Just print out the list of packages for debugging
  pip list
}

install_pip_dependencies() {
  pushd .ci/docker || return
  # Install all Python dependencies, including PyTorch
  pip install --progress-bar off -r requirements-ci.txt
  popd || return
}

install_domains() {
  echo "Install torchvision and torchaudio"
  pip install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${TORCHAUDIO_VERSION}"
  pip install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${TORCHVISION_VERSION}"
}

install_pytorch_and_domains() {
  pushd .ci/docker || return
  TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
  popd || return

  git clone https://github.com/pytorch/pytorch.git

  # Fetch the target commit
  pushd pytorch || return
  git checkout "${TORCH_VERSION}"
  git submodule update --init --recursive

  export USE_DISTRIBUTED=1
  # Then build and install PyTorch
  python setup.py bdist_wheel
  pip install "$(echo dist/*.whl)"

  # Grab the pinned audio and vision commits from PyTorch
  TORCHAUDIO_VERSION=$(cat .github/ci_commit_pins/audio.txt)
  export TORCHAUDIO_VERSION
  TORCHVISION_VERSION=$(cat .github/ci_commit_pins/vision.txt)
  export TORCHVISION_VERSION

  install_domains

  popd || return
  # Print sccache stats for debugging
  sccache --show-stats || true
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
  clean_executorch_install_folders
  mkdir "${CMAKE_OUTPUT_DIR}"

  pushd "${CMAKE_OUTPUT_DIR}" || return
  # This command uses buck2 to gather source files and buck2 could crash flakily
  # on MacOS
  retry cmake -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -DCMAKE_BUILD_TYPE=Release ..
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

cmake_install_executorch_lib() {
  echo "Installing libexecutorch.a and libportable_kernels.a"
  clean_executorch_install_folders
  retry cmake -DBUCK2="$BUCK" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

download_stories_model_artifacts() {
  # Download stories110M.pt and tokenizer from Github
  curl -Ls "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt" --output stories110M.pt
  curl -Ls "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model
  # Create params.json file
  touch params.json
  echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
}

do_not_use_nightly_on_ci() {
  # An assert to make sure that we are not using PyTorch nightly on CI to prevent
  # regression as documented in https://github.com/pytorch/executorch/pull/6564
  TORCH_VERSION=$(pip list | grep -w 'torch ' | awk -F ' ' {'print $2'} | tr -d '\n')

  # The version of PyTorch building from source looks like 2.6.0a0+gitc8a648d that
  # includes the commit while nightly (2.6.0.dev20241019+cpu) or release (2.6.0)
  # won't have that. Note that we couldn't check for the exact commit from the pin
  # ci_commit_pins/pytorch.txt here because the value will be different when running
  # this on PyTorch CI
  if [[ "${TORCH_VERSION}" != *"+git"* ]]; then
    echo "Unexpected torch version. Expected binary built from source, got ${TORCH_VERSION}"
    exit 1
  fi
}
