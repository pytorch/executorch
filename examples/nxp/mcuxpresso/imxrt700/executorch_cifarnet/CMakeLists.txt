# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.10.0)

# THE VERSION NUMBER
set(MCUXPRESSO_CMAKE_FORMAT_MAJOR_VERSION 2)
set(MCUXPRESSO_CMAKE_FORMAT_MINOR_VERSION 0)

set(CMAKE_EXECUTABLE_LIBRARY_PREFIX)
set(CMAKE_EXECUTABLE_LIBRARY_SUFFIX)

# CURRENT DIRECTORY
set(ProjDirPath ${CMAKE_CURRENT_SOURCE_DIR})

set(EXECUTABLE_OUTPUT_PATH ${ProjDirPath}/${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${ProjDirPath}/${CMAKE_BUILD_TYPE})

project(executorch_cifarnet)

enable_language(ASM)

set(MCUX_BUILD_TYPES flash_debug flash_release)

set(MCUX_SDK_PROJECT_NAME executorch_cifarnet.elf)

set(EXECUTORCH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

file(
  GLOB
  example_sources
  ${SdkRootDirPath}/boards/mimxrt700evk/eiq_examples/executorch_cifarnet/cm33_core0/*
)

add_executable(
  ${MCUX_SDK_PROJECT_NAME}
  ${example_sources}
  ${EXECUTORCH_ROOT_DIR}/backends/nxp/runtime/NeutronBackend.cpp
  ${SdkRootDirPath}/middleware/tfm/tf-m/platform/ext/common/syscalls_stub.c
)

add_config_file(
  ${SdkRootDirPath}/devices/MIMXRT798S/mcuxpresso/startup_MIMXRT798S_cm33_core0.c
  ""
  device_startup.MIMXRT798S
)

include(${SdkRootDirPath}/tools/cmake_toolchain_files/mcux_config.cmake)
include(config.cmake)
include(flags.cmake)
include(${SdkRootDirPath}/devices/MIMXRT798S/all_lib_device.cmake)

set(EXECUTORCH_BUILD_PYBIND OFF)
set(EXECUTORCH_BUILD_TESTS OFF)
set(EXECUTORCH_BUILD_DEVTOOLS OFF)
set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
set(EXECUTORCH_BUILD_CPUINFO OFF)
set(EXECUTORCH_BUILD_PTHREADPOOL OFF)
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
set(EXECUTORCH_BUILD_PORTABLE_OPS ON)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
add_subdirectory(${EXECUTORCH_ROOT_DIR} EXCLUDE_FROM_ALL executorch)

target_link_libraries(
  ${MCUX_SDK_PROJECT_NAME}
  PRIVATE
    -Wl,--start-group
    executorch
    executorch_core
    extension_runner_util
    quantized_kernels
    portable_kernels
    ${SdkRootDirPath}/middleware/eiq/neutron/rt700/cm33/libNeutronDriver.a
    ${SdkRootDirPath}/middleware/eiq/neutron/rt700/cm33/libNeutronFirmware.a
    -Wl,--end-group
)

# wrap all libraries with -Wl,--start-group -Wl,--end-group to prevent link
# order issue
group_link_libraries()
