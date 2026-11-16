# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16)
project(nxp_executor_runner C CXX)

option(EVENT_TRACER "Build with event tracing support" ON)

set(NXP_RUNNER_DEFINES NEUTRON_CMODEL)
set(NXP_RUNNER_LIBS)

set(EXECUTORCH_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../)

include(${EXECUTORCH_PATH}/backends/nxp/cmake/neutron_sdk.cmake)

message(STATUS "Looking for Neutron Driver")
if(NOT TARGET nd)
  message(STATUS "Looking for Neutron Driver -- not found")

  set(EIQ_NEUTRON_TARGET_DRIVER
      "${EIQ_NEUTRON_TARGET_DIR}/cmodel/libNeutronDriver.a"
  )
  set(EIQ_NEUTRON_TARGET_FIRMWARE
      "${EIQ_NEUTRON_TARGET_DIR}/cmodel/NeutronFirmware.elf"
  )

  add_library(nd STATIC IMPORTED GLOBAL)
  set_target_properties(
    nd PROPERTIES IMPORTED_LOCATION ${EIQ_NEUTRON_TARGET_DRIVER}
  )
  set_target_properties(
    nd PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                  "${EIQ_NEUTRON_TARGET_INCLUDE_DIR}"
  )

  # Add a virtual target to dump the path to corresponding neutron_firmware
  add_custom_target(locate_neutron_firmware echo ${EIQ_NEUTRON_TARGET_FIRMWARE})
else()
  message(STATUS "Neutron Driver -- found")
endif()

# Check if the ExecuTorch Target is already defined, what indicate this
# CMakeLists.txt is involved as subproject:
message(STATUS "Looking for executorch target")
if(NOT TARGET executorch)
  message(STATUS "Looking for executorch target -- not found, adding")
  set(EXECUTORCH_BUILD_NXP_NEUTRON ON)
  set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
  set(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT ON)
  if(EVENT_TRACER)
    set(EXECUTORCH_BUILD_DEVTOOLS ON)
    set(EXECUTORCH_ENABLE_EVENT_TRACER ON)
  endif()
  add_subdirectory(
    "${EXECUTORCH_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/executorch"
    EXCLUDE_FROM_ALL
  )
else()
  message(STATUS "Looking for executorch target -- found")
endif()

if(EVENT_TRACER)
  list(APPEND NXP_RUNNER_DEFINES ET_EVENT_TRACER_ENABLED)
  list(APPEND NXP_RUNNER_LIBS etdump flatccrt)

  message(STATUS "Looking for flatccrt")
  if(NOT TARGET flatccrt)
    message(STATUS "Looking for flatccrt -- not found, adding")
    add_subdirectory(
      "${EXECUTORCH_PATH}/third-party/flatcc"
      "${CMAKE_CURRENT_BINARY_DIR}/flatcc" EXCLUDE_FROM_ALL
    )
  else()
    message(STATUS "Looking for flatcc -- found")
  endif()
endif()

message(STATUS "Looking for gflags")
if(NOT TARGET gflags::gflags)
  message(STATUS "Looking for gflags -- not found, adding")
  add_subdirectory(
    "${EXECUTORCH_PATH}/third-party/gflags"
    "${CMAKE_CURRENT_BINARY_DIR}/gflags" EXCLUDE_FROM_ALL
  )
else()
  message(STATUS "Looking for gflags -- found")
endif()

set(CMAKE_CXX_STANDARD 17)
add_executable(
  nxp_executor_runner
  nxp_executor_runner.cpp
  ${EXECUTORCH_PATH}/extension/data_loader/file_data_loader.cpp
)

target_compile_definitions(nxp_executor_runner PRIVATE ${NXP_RUNNER_DEFINES})

target_link_libraries(
  nxp_executor_runner
  PRIVATE executorch
          executorch_kernels
          nd
          gflags::gflags
          "-Wl,--whole-archive"
          executorch_delegate_neutron
          "-Wl,--no-whole-archive"
          ${NXP_RUNNER_LIBS}
)
