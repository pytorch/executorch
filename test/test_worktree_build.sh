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
source_entries="$(git -C "${repo_root}" ls-tree --name-only HEAD)"
if [[ -z "${source_entries}" ]]; then
  echo "Failed to list tracked files in ${repo_root}" >&2
  exit 1
fi
while IFS= read -r entry; do
  cmake -E create_symlink "${repo_root}/${entry}" "${source_dir}/${entry}"
done <<< "${source_entries}"

vulkan_source_root="$(
  cd "${source_dir}"
  ./backends/vulkan/test/custom_ops/build_and_run.sh --print-source-root
)"
test "${vulkan_source_root}" -ef "${source_dir}"

nested_source_dir="${test_root}/nested-source"
nested_build_dir="${nested_source_dir}/examples/demo/build"
mkdir -p "${nested_source_dir}/examples/demo" \
  "${nested_source_dir}/examples/other"
printf '#pragma once\n' > "${nested_source_dir}/examples/demo/header.h"
printf '%s\n' \
  'cmake_minimum_required(VERSION 3.19)' \
  'project(nested_build_test NONE)' \
  "include(\"${repo_root}/tools/cmake/Utils.cmake\")" \
  'executorch_get_build_include_dir(' \
  '  "${CMAKE_CURRENT_SOURCE_DIR}" EXECUTORCH_SOURCE_INCLUDE_DIR' \
  ')' > "${nested_source_dir}/CMakeLists.txt"
mkdir -p "${nested_build_dir}/executorch_source_include/executorch"
cmake -E create_symlink "${nested_source_dir}/examples" \
  "${nested_build_dir}/executorch_source_include/executorch/examples"
cmake -S "${nested_source_dir}" -B "${nested_build_dir}"
nested_include_dir="${nested_build_dir}/executorch_source_include/executorch"
test ! -L "${nested_include_dir}/examples"
test ! -L "${nested_include_dir}/examples/demo"
test -f "${nested_include_dir}/examples/demo/header.h"
test ! -e "${nested_include_dir}/examples/demo/build"
test "${nested_include_dir}/examples/other" \
  -ef "${nested_source_dir}/examples/other"
find -L "${nested_include_dir}" -type d -print > /dev/null

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
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
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
"${PYTHON_EXECUTABLE:-python3}" - "${build_dir}/compile_commands.json" \
  "${test_root}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

commands_path = Path(sys.argv[1])
forbidden_path = Path(sys.argv[2]).resolve()
offenders = []

for entry in json.loads(commands_path.read_text()):
    arguments = entry.get("arguments")
    if arguments is None:
        arguments = shlex.split(entry["command"])

    index = 0
    while index < len(arguments):
        argument = arguments[index]
        include_path = None
        if argument in ("-I", "-isystem", "/I"):
            index += 1
            if index < len(arguments):
                include_path = arguments[index]
        elif argument.startswith("-I") and len(argument) > 2:
            include_path = argument[2:]
        elif argument.startswith("-isystem") and len(argument) > 8:
            include_path = argument[8:]
        elif argument.startswith("/I") and len(argument) > 2:
            include_path = argument[2:]

        if include_path is not None:
            resolved_path = Path(include_path)
            if not resolved_path.is_absolute():
                resolved_path = Path(entry["directory"]) / resolved_path
            if resolved_path.resolve() == forbidden_path:
                offenders.append(f"{entry['file']}: {argument}")
        index += 1

if offenders:
    print("Compiler commands include the checkout parent:", file=sys.stderr)
    print("\n".join(offenders), file=sys.stderr)
    raise SystemExit(1)
PY
cmake --build "${build_dir}" --target executorch selective_build --parallel 2
