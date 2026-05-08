#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# build_and_run_tests.sh — self-contained build + run for metal_v2 unit
# tests. Tests run under BOTH MTL4 and MTL3 + MPSGraph configurations
# by default, since the design contract is that behavior is identical
# across the two backend variants.
#
# Design intent
# -------------
# - Self-sufficient: configure + build + run with no external setup
#   beyond a working cmake/make + an Apple toolchain. The script does
#   not assume any pre-existing build directory in the repo.
#
# - Script-owned build dir: defaults to `<this-dir>/.build-<config>`
#   (e.g., `.build-mtl4`, `.build-mtl3`). Hidden so editor / file
#   browser doesn't show it; alongside the script so it's discoverable
#   and easy to clean up. Don't reuse `cmake-out` / `cmake-out-v2` etc.
#   — those are session-specific naming conventions that vary across
#   developers.
#
# - Two configs, one per backend variant: METAL_V2_USE_MTL4=ON and
#   METAL_V2_USE_MTL4=OFF (= MTL3 + MPSGraph). The script builds + runs
#   each in its own build dir; behavior MUST be identical — the single-
#   set design is backend-uniform.
#
# - Idempotent: re-runs are cheap (cmake's incremental build skips
#   unchanged work). Use `--reconfigure` to wipe + re-configure if you
#   want to start from scratch.
#
# Usage
# -----
#   ./build_and_run_tests.sh                  # both configs, all tests
#   ./build_and_run_tests.sh --config mtl4    # only MTL4 build
#   ./build_and_run_tests.sh --config mtl3    # only MTL3 build
#   ./build_and_run_tests.sh --reconfigure    # wipe build dir(s), fresh cmake
#   ./build_and_run_tests.sh -t TARGET        # build + run single target
#                                             # e.g. metal_v2_test_residency_manager
#   ./build_and_run_tests.sh --build-dir DIR  # use custom build dir base
#                                             # (suffixed with -mtl4 / -mtl3)
#
# Environment overrides
# ---------------------
#   BUILD_DIR_BASE  same as --build-dir
#   JOBS            number of parallel build jobs (default: hardware concurrency)
#

set -uo pipefail

#
# Resolve paths.
#
# Script is at <repo>/third-party/executorch/backends/metal/tests/.
# We need:
#   EXECUTORCH_DIR  = <repo>/third-party/executorch  (for cmake -S)
#   SCRIPT_DIR      = <this-dir>
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Default build dir base. Per-config suffix appended below.
DEFAULT_BUILD_DIR_BASE="${SCRIPT_DIR}/.build"

# All test targets defined in this directory's CMakeLists.txt.
TESTS=(
  metal_v2_test_residency_manager
  metal_v2_test_buffer_registry
  metal_v2_test_metal_allocator
  metal_v2_test_metal_buffer_pool
  metal_v2_test_hazard_tracker
  metal_v2_test_metal_command_recorder
  metal_v2_test_metal_kernel_cache
  metal_v2_test_metal_heap
  metal_v2_test_metal_op_registry
  metal_v2_test_metal_device_info
  metal_v2_test_metal_kernel
  metal_v2_test_metal_kernel_compiler
  metal_v2_test_concurrency
  metal_v2_test_mtl3_backend
  metal_v2_test_mps_interop
  metal_v2_test_typed_setters
  metal_v2_test_stream_lifecycle
  metal_v2_test_mtl4_backend
)

#
# Argument parsing.
#
BUILD_DIR_BASE="${BUILD_DIR_BASE:-${DEFAULT_BUILD_DIR_BASE}}"
RECONFIGURE=0
SINGLE_TARGET=""
CONFIG_FILTER="both"  # both | mtl4 | mtl3

print_usage() {
  sed -n '/^# Usage$/,/^# Environment overrides$/p' "${BASH_SOURCE[0]}" \
    | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR_BASE="$2"; shift 2 ;;
    --build-dir=*)
      BUILD_DIR_BASE="${1#*=}"; shift ;;
    --reconfigure)
      RECONFIGURE=1; shift ;;
    -t|--target)
      SINGLE_TARGET="$2"; shift 2 ;;
    --config)
      CONFIG_FILTER="$2"; shift 2 ;;
    --config=*)
      CONFIG_FILTER="${1#*=}"; shift ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      print_usage >&2
      exit 2 ;;
  esac
done

case "${CONFIG_FILTER}" in
  both) CONFIGS=(mtl4 mtl3) ;;
  mtl4) CONFIGS=(mtl4) ;;
  mtl3) CONFIGS=(mtl3) ;;
  *)
    echo "unknown --config: ${CONFIG_FILTER} (expected: both, mtl4, or mtl3)" >&2
    exit 2 ;;
esac

# Default JOBS to hardware concurrency.
if [[ -z "${JOBS:-}" ]]; then
  if command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  elif command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc 2>/dev/null || echo 4)"
  else
    JOBS=4
  fi
fi

# If user filtered to a single target, validate it.
if [[ -n "${SINGLE_TARGET}" ]]; then
  found=0
  for t in "${TESTS[@]}"; do
    [[ "${t}" == "${SINGLE_TARGET}" ]] && found=1 && break
  done
  if [[ ${found} -eq 0 ]]; then
    echo "unknown target: ${SINGLE_TARGET}" >&2
    echo "valid targets:" >&2
    printf '  %s\n' "${TESTS[@]}" >&2
    exit 2
  fi
  TESTS=("${SINGLE_TARGET}")
fi

#
# Per-config build + test loop.
#
# Each config gets its own build dir + cmake configure (with the
# appropriate METAL_V2_USE_MTL4 setting) + build of all targets +
# run of all binaries.
#
ALL_RESULTS=()
ALL_PASS_COUNT=0
ALL_FAIL_COUNT=0

for cfg in "${CONFIGS[@]}"; do
  BUILD_DIR="${BUILD_DIR_BASE}-${cfg}"
  case "${cfg}" in
    mtl4) METAL_V2_USE_MTL4_VALUE=ON ;;
    mtl3) METAL_V2_USE_MTL4_VALUE=OFF ;;
  esac

  echo
  echo "############################################################"
  echo "# Config: ${cfg}  (METAL_V2_USE_MTL4=${METAL_V2_USE_MTL4_VALUE})"
  echo "# Build dir: ${BUILD_DIR}"
  echo "############################################################"
  echo

  # Wipe + reconfigure if --reconfigure was passed, OR if BUILD_DIR
  # has no Makefile (first-time / partial-failed configure).
  needs_configure=0
  if [[ ${RECONFIGURE} -eq 1 ]]; then
    echo "=== --reconfigure: wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    needs_configure=1
  elif [[ ! -f "${BUILD_DIR}/CMakeCache.txt" || ! -f "${BUILD_DIR}/Makefile" ]]; then
    needs_configure=1
  fi

  if [[ ${needs_configure} -eq 1 ]]; then
    echo "=== Configuring cmake at ${BUILD_DIR}"
    if ! cmake -B "${BUILD_DIR}" -S "${EXECUTORCH_DIR}" \
          -DEXECUTORCH_BUILD_METAL=ON \
          -DEXECUTORCH_USE_METAL_V2=ON \
          -DEXECUTORCH_METAL4_ENABLE=ON \
          -DEXECUTORCH_BUILD_TESTS=ON \
          -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
          -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
          -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
          -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
          -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
          -DMETAL_V2_USE_MTL4="${METAL_V2_USE_MTL4_VALUE}" \
          -DCMAKE_BUILD_TYPE=Release; then
      echo "ERROR: cmake configure failed for ${cfg}" >&2
      exit 1
    fi
  fi

  # Build each target. Track per-target build success so we don't
  # later run a stale binary from a previous successful build.
  echo
  echo "=== Building (jobs=${JOBS}, config=${cfg})"
  echo

  BUILD_FAILED=()
  for t in "${TESTS[@]}"; do
    echo "--- Building ${t}"
    if ! cmake --build "${BUILD_DIR}" -j "${JOBS}" --target "${t}"; then
      echo "ERROR: build of ${t} failed (config=${cfg})" >&2
      ALL_RESULTS+=("FAIL-BUILD ${cfg}/${t}")
      ALL_FAIL_COUNT=$((ALL_FAIL_COUNT + 1))
      BUILD_FAILED+=("${t}")
    fi
    echo
  done

  # Helper: return 0 if $1 is in BUILD_FAILED.
  build_failed() {
    local needle="$1"
    [[ ${#BUILD_FAILED[@]} -eq 0 ]] && return 1
    local x
    for x in "${BUILD_FAILED[@]}"; do
      [[ "${x}" == "${needle}" ]] && return 0
    done
    return 1
  }

  # Run each test binary that built successfully.
  echo "=== Running tests (config=${cfg})"
  echo

  for t in "${TESTS[@]}"; do
    if build_failed "${t}"; then
      # Build failed; skip run (already reported FAIL-BUILD above).
      continue
    fi
    BIN="${BUILD_DIR}/backends/metal/tests/${t}"
    if [[ ! -x "${BIN}" ]]; then
      echo "SKIP ${cfg}/${t} (binary not found at ${BIN})"
      ALL_RESULTS+=("SKIP   ${cfg}/${t}")
      continue
    fi

    echo "--- Running ${cfg}/${t}"
    if "${BIN}" --gtest_color=yes; then
      ALL_RESULTS+=("PASS   ${cfg}/${t}")
      ALL_PASS_COUNT=$((ALL_PASS_COUNT + 1))
    else
      ALL_RESULTS+=("FAIL   ${cfg}/${t}")
      ALL_FAIL_COUNT=$((ALL_FAIL_COUNT + 1))
    fi
    echo
  done
done

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
for r in "${ALL_RESULTS[@]}"; do
  echo "  ${r}"
done
echo
echo "  PASS: ${ALL_PASS_COUNT}    FAIL: ${ALL_FAIL_COUNT}"
echo

[[ ${ALL_FAIL_COUNT} -gt 0 ]] && exit 1
exit 0
