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

update_tokenizers_git_submodule() {
  echo "Updating tokenizers git submodule..."
  git submodule update --init
  pushd extension/llm/tokenizers
  git submodule update --init
  popd
}

install_executorch() {
  which pip
  # Install executorch, this assumes that Executorch is checked out in the
  # current directory.
  ./install_executorch.sh --pybind xnnpack "$@"
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

  SYSTEM_NAME=$(uname)
  # The platform version needs to match MACOSX_DEPLOYMENT_TARGET used to build the wheel
  PLATFORM=$(python -c 'import sysconfig; platform=sysconfig.get_platform(); platform[1]="14_0"; print("_".join(platform))')
  PYTHON_VERSION=$(python -c 'import platform; v=platform.python_version_tuple(); print(f"{v[0]}{v[1]}")')
  TORCH_RELEASE=$(cat version.txt)
  TORCH_SHORT_HASH=${TORCH_VERSION:0:7}
  TORCH_WHEEL_PATH="cached_artifacts/pytorch/executorch/pytorch_wheels/${SYSTEM_NAME}/${PYTHON_VERSION}"
  TORCH_WHEEL_NAME="torch-${TORCH_RELEASE}%2Bgit${TORCH_SHORT_HASH}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-${PLATFORM}.whl"

  CACHE_TORCH_WHEEL="https://gha-artifacts.s3.us-east-1.amazonaws.com/${TORCH_WHEEL_PATH}/${TORCH_WHEEL_NAME}"
  # Cache PyTorch wheel is only needed on MacOS, Linux CI already has this as part
  # of the Docker image
  if [[ "${SYSTEM_NAME}" == "Darwin" ]]; then
    pip install "${CACHE_TORCH_WHEEL}" || TORCH_WHEEL_NOT_FOUND=1
  fi

  # Found no such wheel, we will build it from source then
  if [[ "${TORCH_WHEEL_NOT_FOUND:-0}" == "1" ]]; then
    USE_DISTRIBUTED=1 MACOSX_DEPLOYMENT_TARGET=14.0 python setup.py bdist_wheel
    pip install "$(echo dist/*.whl)"

    # Only AWS runners have access to S3
    if command -v aws && [[ -z "${GITHUB_RUNNER:-}" ]]; then
      for WHEEL_PATH in dist/*.whl; do
        WHEEL_NAME=$(basename "${WHEEL_PATH}")
        aws s3 cp "${WHEEL_PATH}" "s3://gha-artifacts/${TORCH_WHEEL_PATH}/${WHEEL_NAME}"
      done
    fi
  else
    echo "Use cached wheel at ${CACHE_TORCH_WHEEL}"
  fi

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
  if [[ $1 == "Debug" ]]; then
      CXXFLAGS="-fsanitize=address,undefined"
  else
      CXXFLAGS=""
  fi
  # This command uses buck2 to gather source files and buck2 could crash flakily
  # on MacOS
  CXXFLAGS="$CXXFLAGS" retry cmake -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -DCMAKE_BUILD_TYPE="${1:-Release}" ..
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
    build_executorch_runner_cmake "$2"
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


parse_args() {
  local args=("$@")
  local i
  local BUILD_TOOL=""
  local BUILD_MODE=""
  local EDITABLE=""
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --build-tool)
        BUILD_TOOL="${args[$((i+1))]}"
        i=$((i+1))
        ;;
      --build-mode)
        BUILD_MODE="${args[$((i+1))]}"
        i=$((i+1))
        ;;
      --editable)
        EDITABLE="${args[$((i+1))]}"
        i=$((i+1))
        ;;
      *)
        echo "Invalid argument: ${args[$i]}"
        exit 1
        ;;
    esac
  done

  if [ -z "$BUILD_TOOL" ]; then
    echo "Missing build tool (require buck2 or cmake), exiting..."
    exit 1
  elif ! [[ $BUILD_TOOL =~ ^(cmake|buck2)$ ]]; then
    echo "Require buck2 or cmake for --build-tool, got ${BUILD_TOOL}, exiting..."
    exit 1
  fi
  BUILD_MODE="${BUILD_MODE:-Release}"
  if ! [[ "$BUILD_MODE" =~ ^(Debug|Release)$ ]]; then
    echo "Unsupported build mode ${BUILD_MODE}, options are Debug or Release."
    exit 1
  fi
  EDITABLE="${EDITABLE:-false}"
  if ! [[ $EDITABLE =~ ^(true|false)$ ]]; then
    echo "Require true or false for --editable, got ${EDITABLE}, exiting..."
    exit 1
  fi

  echo "$BUILD_TOOL $BUILD_MODE $EDITABLE"
}
