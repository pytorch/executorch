#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/executorch-worktree-build.XXXXXX")"
trap 'rm -rf "${test_root}"' EXIT

source_dir="${test_root}/arbitrary-checkout-name"
build_dir="${source_dir}/cmake-out"
mkdir "${source_dir}"
while IFS= read -r entry; do
  cmake -E create_symlink "${repo_root}/${entry}" "${source_dir}/${entry}"
done < <(git -C "${repo_root}" ls-tree --name-only HEAD)

# This sibling would be selected by the old `${EXECUTORCH_ROOT}/..` include
# path. A successful build proves headers come through the build-local alias.
mkdir -p "${test_root}/executorch/runtime/core"
printf '#error "included headers from a sibling checkout"\n' \
  > "${test_root}/executorch/runtime/core/error.h"
mkdir -p "${test_root}/executorch/runtime/platform"
printf '#error "included headers from a sibling checkout"\n' \
  > "${test_root}/executorch/runtime/platform/assert.h"

cmake \
  -S "${source_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_PYBIND=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

test ! -L "${build_dir}/executorch_source_include/executorch"
test "${build_dir}/executorch_source_include/executorch/runtime" \
  -ef "${source_dir}/runtime"
rm "${build_dir}/executorch_source_include/executorch/runtime"
cmake -E create_symlink "${test_root}/missing-runtime" \
  "${build_dir}/executorch_source_include/executorch/runtime"
cmake -S "${source_dir}" -B "${build_dir}"
test "${build_dir}/executorch_source_include/executorch/runtime" \
  -ef "${source_dir}/runtime"
cmake --build "${build_dir}" --target executorch selective_build --parallel 2
