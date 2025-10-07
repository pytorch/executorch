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
  ./install_executorch.sh "$@"
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

  local system_name=$(uname)
  if [[ "${system_name}" == "Darwin" ]]; then
    local platform=$(python -c 'import sysconfig; import platform; v=platform.mac_ver()[0].split(".")[0]; platform=sysconfig.get_platform().split("-"); platform[1]=f"{v}_0"; print("_".join(platform))')
  fi
  local python_version=$(python -c 'import platform; v=platform.python_version_tuple(); print(f"{v[0]}{v[1]}")')
  local torch_release=$(cat version.txt)
  local torch_short_hash=${TORCH_VERSION:0:7}
  local torch_wheel_path="cached_artifacts/pytorch/executorch/pytorch_wheels/${system_name}/${python_version}"
  local torch_wheel_name="torch-${torch_release}%2Bgit${torch_short_hash}-cp${python_version}-cp${python_version}-${platform:-}.whl"

  local cached_torch_wheel="https://gha-artifacts.s3.us-east-1.amazonaws.com/${torch_wheel_path}/${torch_wheel_name}"
  # Cache PyTorch wheel is only needed on MacOS, Linux CI already has this as part
  # of the Docker image
  local torch_wheel_not_found=0
  if [[ "${system_name}" == "Darwin" ]]; then
    pip install "${cached_torch_wheel}" || torch_wheel_not_found=1
  else
    torch_wheel_not_found=1
  fi

  # Found no such wheel, we will build it from source then
  if [[ "${torch_wheel_not_found}" == "1" ]]; then
    echo "No cached wheel found, continue with building PyTorch at ${TORCH_VERSION}"

    git submodule update --init --recursive
    USE_DISTRIBUTED=1 python setup.py bdist_wheel
    pip install "$(echo dist/*.whl)"

    # Only AWS runners have access to S3
    if command -v aws && [[ -z "${GITHUB_RUNNER:-}" ]]; then
      for wheel_path in dist/*.whl; do
        local wheel_name=$(basename "${wheel_path}")
        echo "Caching ${wheel_name}"
        aws s3 cp "${wheel_path}" "s3://gha-artifacts/${torch_wheel_path}/${wheel_name}"
      done
    fi
  else
    echo "Use cached wheel at ${cached_torch_wheel}"
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

  if [[ $1 == "Debug" ]]; then
      CXXFLAGS="-fsanitize=address,undefined"
  else
      CXXFLAGS=""
  fi
  CXXFLAGS="$CXXFLAGS" retry cmake \
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DCMAKE_BUILD_TYPE="${1:-Release}" \
    -B${CMAKE_OUTPUT_DIR} .

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
  build_type="${1:-Release}"
  echo "Installing libexecutorch.a and libportable_kernels.a"
  clean_executorch_install_folders
  retry cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=${build_type} \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config ${build_type}
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
