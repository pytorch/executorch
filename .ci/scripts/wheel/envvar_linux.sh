# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/envvar_base.sh"

# Compile device code for the GPU architectures this release row claims, rather than for
# whichever GPU the builder happens to have. A wheel built with detection alone installs on
# every machine the row covers and then fails when a model runs on a different generation.
source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/cuda_arch_list.sh"
# Check the status rather than only the output. An unrecognised row makes the resolver fail, but
# this script is sourced and does not run under a failing-command shell, so ignoring the status
# would leave both variables unset and let the build fall back to detecting the builder's own GPU.
# That is exactly the outcome this file exists to prevent, and it would ship quietly.
if ! _executorch_cuda_arch="$(executorch_cuda_arch_list_with_ptx)"; then
  echo "could not resolve GPU architectures for CU_VERSION=${CU_VERSION:-unset}" >&2
  exit 1
fi
if [ -n "${_executorch_cuda_arch}" ]; then
  if ! _executorch_cuda_cmake_arch="$(executorch_cuda_cmake_arch_list)"; then
    echo "could not resolve CMake GPU architectures for CU_VERSION=${CU_VERSION:-unset}" >&2
    exit 1
  fi
  # PyTorch's CMake rejects CMAKE_CUDA_ARCHITECTURES and overrides it with OFF, which leaves
  # the build compiling for one detected architecture, so the list has to go through the
  # variable PyTorch reads. Both are set: targets that go through PyTorch's CMake honour the
  # first, and any that do not honour the second.
  export TORCH_CUDA_ARCH_LIST="${_executorch_cuda_arch}"
  export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CUDA_ARCHITECTURES=${_executorch_cuda_cmake_arch}"
  echo "CUDA architectures for this row: ${_executorch_cuda_arch}"
fi
