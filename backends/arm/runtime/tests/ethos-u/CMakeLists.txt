# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(arm_ethos_u_runtime_tests NONE)

enable_testing()

if(NOT EXECUTORCH_ROOT)
  message(
    FATAL_ERROR "EXECUTORCH_ROOT must be set to the ExecuTorch repository root."
  )
endif()
get_filename_component(EXECUTORCH_ROOT "${EXECUTORCH_ROOT}" ABSOLUTE)

set(_ethos_u_runtime_tests_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(_ethos_u_runtime_test_output_dir "${CMAKE_CURRENT_BINARY_DIR}/test_run")

message(
  STATUS
    "Ethos-U runtime tests configured. List tests with: ctest --test-dir ${CMAKE_CURRENT_BINARY_DIR} -N"
)
message(
  STATUS
    "Run all tests with output on failure: ctest --test-dir ${CMAKE_CURRENT_BINARY_DIR} --output-on-failure"
)
message(
  STATUS
    "Run one test: ctest --test-dir ${CMAKE_CURRENT_BINARY_DIR} -R '^memory_allocation_planned_fast$' --output-on-failure"
)

add_test(
  NAME memory_allocation_baseline
  COMMAND "${_ethos_u_runtime_tests_dir}/test_memory_allocation_baseline.sh"
          "${EXECUTORCH_ROOT}" "${_ethos_u_runtime_test_output_dir}"
)
set_tests_properties(
  memory_allocation_baseline
  PROPERTIES LABELS "arm;ethos-u;runtime;memory_allocation;fvp" RESOURCE_LOCK
             arm_ethos_u_runtime
)

add_test(
  NAME runner_size_optimization
  COMMAND "${_ethos_u_runtime_tests_dir}/test_runner_size_optimization.sh"
          "${EXECUTORCH_ROOT}" "${_ethos_u_runtime_test_output_dir}"
)
set_tests_properties(
  runner_size_optimization
  PROPERTIES LABELS "arm;ethos-u;runtime;memory_allocation;runner"
             RESOURCE_LOCK arm_ethos_u_runtime
)

add_test(
  NAME memory_allocation_planned_fast
  COMMAND "${_ethos_u_runtime_tests_dir}/test_memory_allocation_planned_fast.sh"
          "${EXECUTORCH_ROOT}" "${_ethos_u_runtime_test_output_dir}"
)
set_tests_properties(
  memory_allocation_planned_fast
  PROPERTIES LABELS "arm;ethos-u;runtime;memory_allocation;fvp" RESOURCE_LOCK
             arm_ethos_u_runtime
)
