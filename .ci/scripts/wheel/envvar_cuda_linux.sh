# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/envvar_base.sh"

# Ask for the CUDA delegate explicitly rather than letting the build detect a toolkit. A detected
# build is fine locally, but a release row states what it is producing, and a row that silently
# produced a CPU wheel because the toolkit was missing would publish under a CUDA name.
export EXECUTORCH_BUILD_CUDA=1
export CMAKE_ARGS="${CMAKE_ARGS} -DEXECUTORCH_BUILD_CUDA=ON"

# Fail the build if CUDA is not actually present. Without this the packaging step would look for
# CUDA libraries that were never built and report a confusing missing-file error several minutes
# after the real problem.
# A regular file, executable, and able to answer: a directory also passes an execute-bit test, and a
# stub or a broken wrapper passes both, which would let the row report a toolkit it cannot compile with.
_executorch_nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
if [ ! -f "${_executorch_nvcc}" ] || [ ! -x "${_executorch_nvcc}" ] ||
  ! "${_executorch_nvcc}" --version >/dev/null 2>&1; then
  echo "EXECUTORCH_BUILD_CUDA is set but ${_executorch_nvcc} is not a working nvcc. This row cannot build a CUDA wheel." >&2
  exit 1
fi

# Compile device code for the GPUs this release row claims, rather than for whichever GPU the
# builder happens to have. A wheel built by detection alone installs on every machine the row covers
# and then fails when a model runs on a different generation.
source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/cuda_arch_list.sh"
# The status is checked rather than only the output, so an unrecognised row reports why it stopped.
# A bare assignment would end the build on the lookup's own exit status with no message, since this
# file is sourced into a shell that exits on a failing command.
if ! _executorch_cuda_arch="$(executorch_cuda_arch_list)"; then
  echo "could not resolve GPU architectures for CU_VERSION=${CU_VERSION:-unset}" >&2
  exit 1
fi
if [ -n "${_executorch_cuda_arch}" ]; then
  export TORCH_CUDA_ARCH_LIST="${_executorch_cuda_arch}"
  echo "building device code for: ${TORCH_CUDA_ARCH_LIST}"
fi
