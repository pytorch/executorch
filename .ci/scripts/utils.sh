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
  # current directory. The --extra-index-url options tell pip to look on the
  # pytorch servers for nightly and pre-release versions of torch packages.
  pip install . --no-build-isolation -v \
      --extra-index-url https://download.pytorch.org/whl/test/cpu \
      --extra-index-url https://download.pytorch.org/whl/nightly/cpu
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

  export _GLIBCXX_USE_CXX11_ABI=0
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
  rm -rf "${CMAKE_OUTPUT_DIR}" && mkdir "${CMAKE_OUTPUT_DIR}"

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
  rm -rf cmake-out
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

install_swiftshader() {
  _https_amazon_aws=https://ossci-android.s3.amazonaws.com
  _swiftshader_archive=swiftshader-abe07b943-prebuilt.tar.gz
  _swiftshader_dir=/var/lib/swiftshader
  mkdir -p $_swiftshader_dir

  _tmp_archive="/tmp/${_swiftshader_archive}"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "$_https_amazon_aws/${_swiftshader_archive}"

  tar -C "${_swiftshader_dir}" -xzf "${_tmp_archive}"

  export VK_ICD_FILENAMES="${_swiftshader_dir}/build/Linux/vk_swiftshader_icd.json"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_swiftshader_dir}/build/Linux/libvulkan.so.1"
}

install_glslc() {
  _shaderc_url_base="https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/linux/continuous_clang_release"
  _shaderc_version="448/20240305-065535"
  _shaderc_url="${_shaderc_url_base}/${_shaderc_version}/install.tgz"

  _glslc_dir=/var/lib/shaderc
  mkdir -p $_glslc_dir

  _tmp_archive="/tmp/install.tgz"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "${_shaderc_url}"

  tar -C "${_glslc_dir}" -xzf "${_tmp_archive}"

  export PATH="${PATH}:${_glslc_dir}/install/bin/"
}
