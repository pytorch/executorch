#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Cross-compile cadence_executor_runner for a Cadence Xtensa core and (by
# default) smoke-test it on the Instruction Set Simulator with a trivial model.
#
# Requires the Xtensa toolchain env to already be set (run
# .ci/scripts/setup-xtensa-tools.sh <backend> first): XTENSA_TOOLCHAIN,
# TOOLCHAIN_VER, XTENSA_SYSTEM, XTENSA_CORE, XTENSAD_LICENSE_FILE,
# CADENCE_OPT_FLAG, and xt-clang on PATH.
#
# Usage:
#   .ci/scripts/build-cadence-xtensa.sh [--no-run]
#     --no-run : compile only, skip the ISS smoke test

set -euo pipefail

RUN_SMOKE=1
[[ "${1:-}" == "--no-run" ]] && RUN_SMOKE=0

: "${XTENSA_TOOLCHAIN:?run setup-xtensa-tools.sh first}"
: "${TOOLCHAIN_VER:?run setup-xtensa-tools.sh first}"
: "${XTENSA_CORE:?run setup-xtensa-tools.sh first}"
: "${CADENCE_OPT_FLAG:?run setup-xtensa-tools.sh first}"

# [do-not-land] Stand in for an mt-l-x86iavx512-8-64 runner, whose 48-character
# pod hostname is what breaks the RJ-2025.5 licence key. Takes that exact name,
# shows which key xt-clang then asks for, and falls through to the shim, which
# should recover and let the build run.
if [[ -z "${XTENSA_FAKE_LONG_HOSTNAME:-}" ]]; then
  export XTENSA_FAKE_LONG_HOSTNAME=1
  exec unshare -Uur bash -c '
    hostname mt-l-x86iavx512-8-64-r6gdl-runner-6525q-workflow
    _hn=$(hostname); echo "=== posing as an avx512 pod: ${_hn} (${#_hn} chars) ==="
    echo "int main(void){return 0;}" > /tmp/_lic_probe.c
    echo -n "  at this hostname xt-clang asks for: "
    FLEXLM_DIAGNOSTICS=3 xt-clang -c /tmp/_lic_probe.c -o /tmp/_lic_probe.o 2>&1 \
      | grep -oE "(Checkout succeeded: )?XT_XCC_TIE[A-Z0-9_]*" | head -1
    exec "$0" "$@"' "$0" "$@"
fi

# shellcheck source=.ci/scripts/xtensa-short-hostname.sh
source "$(dirname "${BASH_SOURCE[0]}")/xtensa-short-hostname.sh"

NPROC=$(nproc)
echo "=== building cadence_executor_runner for ${XTENSA_CORE} (${CADENCE_OPT_FLAG}) ==="
xt-clang --version | head -1

rm -rf cmake-out
CXXFLAGS="-fno-exceptions -fno-rtti" cmake \
  -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_CADENCE=ON \
  "-D${CADENCE_OPT_FLAG}=ON" \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_CADENCE_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DEXECUTORCH_USE_DL=OFF \
  -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
  -DEXECUTORCH_BUILD_DEVTOOLS=OFF \
  -DHAVE_FNMATCH_H=OFF \
  -DFLATCC_ALLOW_WERROR=OFF \
  -DPYTHON_EXECUTABLE="$(which python3)" \
  -Bcmake-out .

cmake --build cmake-out --target cadence_executor_runner -j"${NPROC}"

RUNNER="cmake-out/backends/cadence/cadence_executor_runner"
if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: ${RUNNER} was not produced" >&2
  exit 1
fi
command -v file >/dev/null 2>&1 && file "${RUNNER}" || true
echo "Build OK: ${RUNNER}"

if [[ "${RUN_SMOKE}" == "0" ]]; then
  echo "Skipping ISS smoke test (--no-run)."
  exit 0
fi

echo "=== ISS smoke test: export add.pte and run on xt-run --turbo ==="
python3 -m examples.portable.scripts.export --model_name=add >/dev/null
LOG=$(mktemp)
xt-run --turbo "${RUNNER}" --model_path=add.pte 2>&1 | tee "${LOG}"
if ! grep -q "Model executed successfully" "${LOG}"; then
  echo "ERROR: ISS smoke test did not report success for ${XTENSA_CORE}" >&2
  exit 1
fi
echo "ISS smoke test passed for ${XTENSA_CORE}."
