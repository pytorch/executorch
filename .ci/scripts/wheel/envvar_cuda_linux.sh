# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is sourced into the environment before building a pip wheel. It
# should typically only contain shell variable assignments. Be sure to export
# any variables so that subprocesses will see them.

source "${GITHUB_WORKSPACE}/${REPOSITORY}/.ci/scripts/wheel/envvar_base.sh"

# Ask for the CUDA delegate explicitly rather than letting the build detect a toolkit. A
# detected build is fine locally, but a release row states what it is producing, and a row
# that silently produced a CPU wheel because the toolkit was not found would publish under a
# CUDA name.
export EXECUTORCH_BUILD_CUDA=1
export CMAKE_ARGS="${CMAKE_ARGS} -DEXECUTORCH_BUILD_CUDA=ON"

# Fail the build if CUDA is not actually present. Without this the packaging step would look
# for CUDA libraries that were never built and report a confusing missing-file error, several
# minutes after the real problem.
if [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
  echo "EXECUTORCH_BUILD_CUDA is set but no nvcc was found. This row cannot build a CUDA wheel." >&2
  exit 1
fi
