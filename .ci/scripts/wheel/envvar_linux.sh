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
_executorch_cuda_arch="$(executorch_cuda_arch_list)"
if [ -n "${_executorch_cuda_arch}" ]; then
  export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CUDA_ARCHITECTURES=${_executorch_cuda_arch}"
  echo "CUDA architectures for this row: ${_executorch_cuda_arch}"
fi
