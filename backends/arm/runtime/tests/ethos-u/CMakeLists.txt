# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(arm_ethos_u_runtime_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

if(NOT EXECUTORCH_ROOT)
  message(
    FATAL_ERROR "EXECUTORCH_ROOT must be set to the ExecuTorch repository root."
  )
endif()
get_filename_component(EXECUTORCH_ROOT "${EXECUTORCH_ROOT}" ABSOLUTE)

add_subdirectory(
  "${EXECUTORCH_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/executorch"
  EXCLUDE_FROM_ALL
)

include("${EXECUTORCH_ROOT}/backends/arm/scripts/corstone_utils.cmake")
set(ETHOS_SDK_PATH
    "${EXECUTORCH_ROOT}/examples/arm/arm-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)
option(FETCH_ETHOS_U_CONTENT "Fetch Ethos-U dependencies" ON)
arm_ensure_ethos_u_content(
  "${ETHOS_SDK_PATH}" "${EXECUTORCH_ROOT}" ${FETCH_ETHOS_U_CONTENT} DRIVER_ONLY
)

find_package(GTest QUIET)
if(NOT TARGET GTest::gtest_main)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE
  )
  add_subdirectory(
    "${EXECUTORCH_ROOT}/third-party/googletest"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL
  )
endif()

add_executable(
  ethosu_io_plan_test
  test_ethosu_io.cpp "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUIOPlan.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_IoMemcpy.cpp"
)
target_link_libraries(
  ethosu_io_plan_test PRIVATE executorch_core GTest::gtest_main
)
add_test(NAME ethosu_io_plan_test COMMAND ethosu_io_plan_test)
set_tests_properties(
  ethosu_io_plan_test PROPERTIES LABELS "arm;ethos-u;runtime"
)

add_executable(
  ethosu_planned_scratch_test
  test_ethosu_planned_scratch.cpp
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUIOPlan.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_IoMemcpy.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/VelaBinStream.cpp"
)
target_link_libraries(
  ethosu_planned_scratch_test PRIVATE executorch_core GTest::gtest_main
)
add_test(NAME ethosu_planned_scratch_test COMMAND ethosu_planned_scratch_test)
set_tests_properties(
  ethosu_planned_scratch_test PROPERTIES LABELS "arm;ethos-u;runtime"
)

add_executable(
  ethosu_scratch_io_test
  test_ethosu_scratch_io.cpp
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUIOPlan.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_IoMemcpy.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/VelaBinStream.cpp"
)
target_link_libraries(
  ethosu_scratch_io_test PRIVATE executorch_core GTest::gtest_main
)
add_test(NAME ethosu_scratch_io_test COMMAND ethosu_scratch_io_test)
set_tests_properties(
  ethosu_scratch_io_test PROPERTIES LABELS "arm;ethos-u;runtime"
)

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

add_executable(
  vela_external_blocks_test
  "${EXECUTORCH_ROOT}/backends/arm/runtime/VelaBinStream.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/test/vela_external_blocks_test.cpp"
)
target_link_libraries(
  vela_external_blocks_test PRIVATE executorch_core GTest::gtest_main
)
add_test(NAME vela_external_blocks COMMAND vela_external_blocks_test)
set_tests_properties(
  vela_external_blocks PROPERTIES LABELS "arm;ethos-u;runtime"
)

add_executable(
  ethosu_backend_init_test
  test_ethosu_backend_init.cpp
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUIOPlan.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_IoMemcpy.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/VelaBinStream.cpp"
)
target_link_libraries(
  ethosu_backend_init_test PRIVATE executorch_core GTest::gtest_main
)
add_test(NAME ethosu_backend_init_test COMMAND ethosu_backend_init_test)
set_tests_properties(
  ethosu_backend_init_test PROPERTIES LABELS "arm;ethos-u;runtime"
)

find_package(Threads REQUIRED)
add_library(
  ethosu_execution_stats_test_library STATIC
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUExecutionStats.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUExecutionStats_CoreDriver.cpp"
)
target_include_directories(
  ethosu_execution_stats_test_library
  PRIVATE "${EXECUTORCH_ROOT}/.."
          "${ETHOS_SDK_PATH}/core_software/core_driver/include"
)
target_compile_definitions(
  ethosu_execution_stats_test_library PRIVATE ET_EVENT_TRACER_ENABLED
)
target_link_libraries(
  ethosu_execution_stats_test_library PUBLIC executorch_core
)

add_library(
  ethosu_execution_stats_no_trace_test_library STATIC
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUExecutionStats.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUExecutionStats_CoreDriver.cpp"
)
target_include_directories(
  ethosu_execution_stats_no_trace_test_library
  PRIVATE "${EXECUTORCH_ROOT}/.."
          "${ETHOS_SDK_PATH}/core_software/core_driver/include"
)
target_link_libraries(
  ethosu_execution_stats_no_trace_test_library PUBLIC executorch_core
)

add_executable(
  ethosu_execution_stats_test
  "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_pmu_test_stubs.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_execution_stats_test.cpp"
)
target_include_directories(
  ethosu_execution_stats_test
  PRIVATE "${EXECUTORCH_ROOT}/.."
          "${ETHOS_SDK_PATH}/core_software/core_driver/include"
)
target_link_libraries(
  ethosu_execution_stats_test
  PRIVATE ethosu_execution_stats_no_trace_test_library Threads::Threads
)
add_test(NAME ethosu_execution_stats COMMAND ethosu_execution_stats_test)
set_tests_properties(
  ethosu_execution_stats PROPERTIES LABELS "arm;ethos-u;runtime"
)

add_executable(
  ethosu_pmu_callbacks_test
  "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_pmu_test_stubs.cpp"
  "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_pmu_callbacks_test.cpp"
)
target_include_directories(
  ethosu_pmu_callbacks_test
  PRIVATE "${EXECUTORCH_ROOT}/.."
          "${ETHOS_SDK_PATH}/core_software/core_driver/include"
)
target_compile_definitions(
  ethosu_pmu_callbacks_test PRIVATE ET_EVENT_TRACER_ENABLED
)
target_link_libraries(ethosu_pmu_callbacks_test PRIVATE Threads::Threads)
executorch_target_whole_archive(
  ethosu_pmu_callbacks_test ethosu_execution_stats_test_library
)
add_test(NAME ethosu_pmu_callbacks COMMAND ethosu_pmu_callbacks_test)
set_tests_properties(
  ethosu_pmu_callbacks PROPERTIES LABELS "arm;ethos-u;runtime"
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
  NAME runner_input_restoration
  COMMAND "${_ethos_u_runtime_tests_dir}/test_runner_input_restoration.sh"
          "${EXECUTORCH_ROOT}" "${_ethos_u_runtime_test_output_dir}"
)
set_tests_properties(
  runner_input_restoration PROPERTIES LABELS "arm;ethos-u;runtime;runner;fvp"
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

set(ETHOSU_LINUX_DRIVER_SOURCE_DIR
    "${EXECUTORCH_ROOT}/examples/arm/arm-scratch/ethos-u/linux_driver_stack"
    CACHE PATH "Linux driver source used by the PMU adapter tests"
)
if(EXISTS
   "${ETHOSU_LINUX_DRIVER_SOURCE_DIR}/kernel/include/direct/interface/ethosu_direct_interface_u65.h"
)
  include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmEthosUPmu.cmake)
  function(add_linux_stats_test_library TARGET)
    add_library(
      ${TARGET} STATIC
      "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUExecutionStats.cpp"
      "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUExecutionStats_LinuxDriver.cpp"
    )
    target_include_directories(
      ${TARGET}
      PUBLIC "${EXECUTORCH_ROOT}/.."
             "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_linux_driver_stub"
             "${ETHOS_SDK_PATH}/core_software/core_driver/include"
    )
    target_compile_definitions(
      ${TARGET} PUBLIC EXECUTORCH_BUILD_ARM_ETHOSU_LINUX=1 ${ARGN}
    )
    target_link_libraries(${TARGET} PUBLIC executorch_core)
    arm_add_ethosu_linux_pmu_events(
      ${TARGET} "${ETHOSU_LINUX_DRIVER_SOURCE_DIR}"
    )
  endfunction()
  add_linux_stats_test_library(
    ethosu_linux_execution_stats_test_library ET_EVENT_TRACER_ENABLED
  )
  add_linux_stats_test_library(ethosu_linux_no_trace_test_library)
  add_executable(
    ethosu_linux_execution_stats_test
    "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_linux_execution_stats_test.cpp"
    "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_LinuxDriver.cpp"
    "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUIOPlan.cpp"
    "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_IoMemcpy.cpp"
  )
  target_link_libraries(
    ethosu_linux_execution_stats_test PRIVATE GTest::gtest_main
  )
  executorch_target_whole_archive(
    ethosu_linux_execution_stats_test ethosu_linux_execution_stats_test_library
  )
  add_test(NAME ethosu_linux_execution_stats
           COMMAND ethosu_linux_execution_stats_test
  )

  foreach(stats_mode IN ITEMS linked unlinked)
    add_executable(
      ethosu_linux_stats_${stats_mode}_test
      "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_linux_optional_stats_test.cpp"
      "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_LinuxDriver.cpp"
      "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUIOPlan.cpp"
      "${EXECUTORCH_ROOT}/backends/arm/runtime/EthosUBackend_IoMemcpy.cpp"
    )
    target_include_directories(
      ethosu_linux_stats_${stats_mode}_test
      PRIVATE "${EXECUTORCH_ROOT}/.."
              "${EXECUTORCH_ROOT}/backends/arm/test/ethosu_linux_driver_stub"
    )
    target_compile_definitions(
      ethosu_linux_stats_${stats_mode}_test
      PRIVATE EXECUTORCH_BUILD_ARM_ETHOSU_LINUX=1
    )
    target_link_libraries(
      ethosu_linux_stats_${stats_mode}_test PRIVATE executorch_core
                                                    GTest::gtest_main
    )
    if(stats_mode STREQUAL "linked")
      target_compile_definitions(
        ethosu_linux_stats_linked_test PRIVATE ET_ARM_ETHOSU_TEST_STATS_LINKED
      )
      executorch_target_whole_archive(
        ethosu_linux_stats_linked_test ethosu_linux_no_trace_test_library
      )
    endif()
    add_test(NAME ethosu_linux_stats_${stats_mode}
             COMMAND ethosu_linux_stats_${stats_mode}_test
    )
  endforeach()
else()
  message(
    STATUS
      "Skipping Linux PMU tests: set ETHOSU_LINUX_DRIVER_SOURCE_DIR to a driver checkout"
  )
endif()

add_test(
  NAME ethosu_scratch_to_executorch
  COMMAND "${_ethos_u_runtime_tests_dir}/test_ethosu_scratch_to_executorch.sh"
          "${EXECUTORCH_ROOT}" "${_ethos_u_runtime_test_output_dir}"
)
set_tests_properties(
  ethosu_scratch_to_executorch
  PROPERTIES LABELS "arm;ethos-u;runtime;memory_allocation;fvp" RESOURCE_LOCK
             arm_ethos_u_runtime
)
