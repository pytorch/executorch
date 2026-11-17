#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if ! command -v cmake > /dev/null; then
  printf 'SKIP: CMake is unavailable\n'
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/vulkan_cmake_tests.XXXXXX")"
CASE_NAME=""
PASSED=0

cleanup() {
  local status=$?
  set +e
  if [ "$status" -ne 0 ]; then
    printf 'FAIL: %s\n' "$CASE_NAME" >&2
    cat "$TEST_TMP/cmake.log" >&2
  fi
  rm -rf "$TEST_TMP"
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$TEST_TMP/producer" "$TEST_TMP/consumer"
printf 'void unused() {}\n' > "$TEST_TMP/producer/empty.cpp"
printf 'int main() { return 0; }\n' > "$TEST_TMP/consumer/main.cpp"

cat > "$TEST_TMP/producer/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.24)
project(producer LANGUAGES CXX)
set(targets executorch executorch_core portable_kernels)
if(TEST_WITH_VULKAN)
  list(APPEND targets vulkan_backend)
endif()
foreach(target IN LISTS targets)
  add_library(${target} STATIC empty.cpp)
endforeach()
if(TEST_WITH_BOOST)
  add_library(Boost::stacktrace_addr2line INTERFACE IMPORTED)
  target_link_libraries(vulkan_backend PRIVATE Boost::stacktrace_addr2line)
endif()
install(TARGETS ${targets} EXPORT ExecuTorchTargets ARCHIVE DESTINATION lib)
install(EXPORT ExecuTorchTargets DESTINATION lib/cmake/ExecuTorch)
install(FILES "${EXECUTORCH_CONFIG}" DESTINATION lib/cmake/ExecuTorch)
CMAKE

cat > "$TEST_TMP/consumer/FindBoost.cmake" <<'CMAKE'
if(NOT Boost_FIND_VERSION STREQUAL "1.89")
  message(FATAL_ERROR "Unexpected Boost version: ${Boost_FIND_VERSION}")
endif()
if(NOT Boost_FIND_COMPONENTS STREQUAL "stacktrace_basic;stacktrace_addr2line")
  message(FATAL_ERROR "Unexpected Boost components: ${Boost_FIND_COMPONENTS}")
endif()
set(Boost_FOUND TRUE)
foreach(component IN LISTS Boost_FIND_COMPONENTS)
  add_library(Boost::${component} INTERFACE IMPORTED)
endforeach()
file(WRITE "${CMAKE_BINARY_DIR}/boost-discovered" "found")
CMAKE

cat > "$TEST_TMP/consumer/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.24)
project(consumer LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(executorch CONFIG REQUIRED PATHS "${TEST_INSTALL}" NO_DEFAULT_PATH)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE executorch executorch_core portable_kernels)
if(TEST_WITH_VULKAN)
  target_link_libraries(consumer PRIVATE vulkan_backend)
endif()
CMAKE

while read -r CASE_NAME with_vulkan with_boost; do
  case_dir="$TEST_TMP/$CASE_NAME"
  disable_boost=ON
  if [ "$with_boost" = ON ]; then disable_boost=OFF; fi

  cmake -S "$TEST_TMP/producer" -B "$case_dir/producer-build" \
    "-DCMAKE_INSTALL_PREFIX=$case_dir/install" \
    "-DEXECUTORCH_CONFIG=$ROOT/tools/cmake/executorch-config.cmake" \
    "-DTEST_WITH_VULKAN=$with_vulkan" "-DTEST_WITH_BOOST=$with_boost" \
    > "$TEST_TMP/cmake.log" 2>&1
  cmake --build "$case_dir/producer-build" --target install --parallel 2 \
    >> "$TEST_TMP/cmake.log" 2>&1
  cmake -S "$TEST_TMP/consumer" -B "$case_dir/consumer-build" \
    "-DTEST_INSTALL=$case_dir/install" "-DTEST_WITH_VULKAN=$with_vulkan" \
    "-DCMAKE_DISABLE_FIND_PACKAGE_Boost=$disable_boost" \
    >> "$TEST_TMP/cmake.log" 2>&1

  if [ "$with_boost" = ON ]; then
    test -f "$case_dir/consumer-build/boost-discovered"
  else
    test ! -f "$case_dir/consumer-build/boost-discovered"
  fi
  cmake --build "$case_dir/consumer-build" --parallel 2 \
    >> "$TEST_TMP/cmake.log" 2>&1
  printf 'PASS: %s\n' "$CASE_NAME"
  PASSED=$((PASSED + 1))
done <<'CASES'
vulkan_with_boost ON ON
vulkan_without_boost ON OFF
without_vulkan OFF OFF
CASES

test "$PASSED" -eq 3
printf '%s passed, 0 failed\n' "$PASSED"
