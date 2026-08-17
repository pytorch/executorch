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
if [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
  echo "EXECUTORCH_BUILD_CUDA is set but no nvcc was found. This row cannot build a CUDA wheel." >&2
  exit 1
fi

# Compile device code for the GPUs this release row claims, rather than for whichever GPU the
# builder happens to have. A wheel built by detection alone installs on every machine the row covers
# and then fails when a model runs on a different generation.
source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/cuda_arch_list.sh"
# The status is checked rather than only the output. An unrecognised row makes the lookup fail, and
# this file is sourced rather than run under a failing-command shell, so ignoring the status would
# leave the variable unset and let the build fall back to detecting the builder's own GPU. That is
# exactly the outcome this is meant to prevent, and it would ship quietly.
if ! _executorch_cuda_arch="$(executorch_cuda_arch_list_with_ptx)"; then
  echo "could not resolve GPU architectures for CU_VERSION=${CU_VERSION:-unset}" >&2
  exit 1
fi
if [ -n "${_executorch_cuda_arch}" ]; then
  export TORCH_CUDA_ARCH_LIST="${_executorch_cuda_arch}"
  echo "building device code for: ${TORCH_CUDA_ARCH_LIST}"
fi
