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

dedupe_macos_loader_path_rpaths() {
  if [[ "$(uname)" != "Darwin" ]]; then
    return
  fi

  local torch_lib_dir
  pushd ..
  torch_lib_dir=$(python -c "import importlib.util; print(importlib.util.find_spec('torch').submodule_search_locations[0])")/lib
  popd

  if [[ -z "${torch_lib_dir}" || ! -d "${torch_lib_dir}" ]]; then
    return
  fi

  local torch_libs=(
    "libtorch_cpu.dylib"
    "libtorch.dylib"
    "libc10.dylib"
  )

  for lib_name in "${torch_libs[@]}"; do
    local lib_path="${torch_lib_dir}/${lib_name}"
    if [[ ! -f "${lib_path}" ]]; then
      continue
    fi

    local removed=0
    # Repeatedly remove the @loader_path rpath entries until none remain.
    while install_name_tool -delete_rpath @loader_path "${lib_path}" 2>/dev/null; do
      removed=1
    done

    if [[ "${removed}" == "1" ]]; then
      install_name_tool -add_rpath @loader_path "${lib_path}" || true
    fi
  done
}

install_pytorch_and_domains() {
  # CWD is the executorch repo root, where torch_pin.py lives.
  local torch_channel
  local torch_spec
  local torchvision_spec
  local torchaudio_spec
  local torch_index_url
  local torch_cache_args_output
  torch_channel=$(python -c "from torch_pin import CHANNEL; print(CHANNEL)")
  torch_spec=$(python -c "from torch_pin import torch_spec; print(torch_spec())")
  torchvision_spec=$(python -c "from torch_pin import torchvision_spec; print(torchvision_spec())")
  torchaudio_spec=$(python -c "from torch_pin import torchaudio_spec; print(torchaudio_spec())")
  torch_index_url=$(python -c "from torch_pin import torch_index_url_base; print(torch_index_url_base())")
  torch_cache_args_output=$(python -c "from torch_pin import pip_cache_args; print(' '.join(pip_cache_args()))")
  local torch_cache_args=()
  if [[ -n "${torch_cache_args_output}" ]]; then
    read -r -a torch_cache_args <<< "${torch_cache_args_output}"
  fi

  local wheelhouse
  wheelhouse=$(mktemp -d)

  local system_name
  local system_arch
  local python_version
  system_name=$(uname)
  system_arch=$(uname -m)
  python_version=$(python -c 'import platform; v=platform.python_version_tuple(); print(f"{v[0]}{v[1]}")')
  local torch_wheel_cache_path="cached_artifacts/pytorch/executorch/pytorch_wheels/${system_name}/${system_arch}/${python_version}/cpu/${torch_channel}"
  local torch_wheel_cache_uri="s3://gha-artifacts/${torch_wheel_cache_path}"

  # Do not cache test-channel wheels in S3: RC artifacts may be re-uploaded
  # under the same package version.
  if [[ "${torch_channel}" != "test" ]] && command -v aws >/dev/null 2>&1; then
    aws s3 sync "${torch_wheel_cache_uri}" "${wheelhouse}" || true
  fi

  python -m pip download --no-deps "${torch_cache_args[@]}" \
    --dest "${wheelhouse}" \
    --find-links "${wheelhouse}" \
    "${torch_spec}" "${torchvision_spec}" "${torchaudio_spec}" \
    --index-url "${torch_index_url}/cpu"

  if [[ "${torch_channel}" != "test" && -z "${GITHUB_RUNNER:-}" ]] && command -v aws >/dev/null 2>&1; then
    aws s3 sync "${wheelhouse}" "${torch_wheel_cache_uri}" \
      --exclude "*" --include "*.whl" || true
  fi

  pip install --force-reinstall "${torch_cache_args[@]}" \
    --find-links "${wheelhouse}" \
    "${torch_spec}" "${torchvision_spec}" "${torchaudio_spec}" \
    --index-url "${torch_index_url}/cpu"
  dedupe_macos_loader_path_rpaths
  rm -rf "${wheelhouse}"
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

  local build_type="${1:-Release}"
  local sanitizer_flag=""

  if [[ "${EXECUTORCH_USE_SANITIZER:-OFF}" == "ON" ]]; then
      sanitizer_flag="-DEXECUTORCH_USE_SANITIZER=ON"
  fi

  retry cmake \
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    ${sanitizer_flag} \
    ${CMAKE_ARGS:-} \
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

verify_torch_matches_pin_on_ci() {
  local expected_torch_version
  local installed_torch_version
  expected_torch_version=$(python - <<'PY'
from torch_pin import CHANNEL, NIGHTLY_VERSION, TORCH_VERSION

if CHANNEL == "nightly":
    print(f"{TORCH_VERSION}.{NIGHTLY_VERSION}")
else:
    print(TORCH_VERSION)
PY
)
  installed_torch_version=$(python - <<'PY'
import torch

print(torch.__version__.split("+", 1)[0])
PY
)

  if [[ "${installed_torch_version}" != "${expected_torch_version}" ]]; then
    echo "Unexpected torch version. Expected ${expected_torch_version}, got ${installed_torch_version}"
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
