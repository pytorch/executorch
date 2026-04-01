# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16)
project(nxp_executor_runner C CXX)

set(EIQ_NEUTRON_TARGET
    "imxrt700"
    CACHE STRING "Neutron Target to build the executor_runner"
)
set(EXECUTORCH_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

message(STATUS "Looking for Neutron Driver")
if(NOT TARGET nd)
  message(STATUS "Looking for Neutron Driver -- not found")

  message(STATUS "Looking for eIQ Neutron SDK")
  execute_process(
    COMMAND
      ${Python_EXECUTABLE} -c
      "import eiq_neutron_sdk, os; print(os.path.dirname(eiq_neutron_sdk.__file__))"
    OUTPUT_VARIABLE EIQ_NEUTRON_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(
    STATUS "Looking for eIQ Neutron SDK -- found: ${EIQ_NEUTRON_SDK_PATH}"
  )
  message(STATUS "Looking for Neutron target $CACHE{EIQ_NEUTRON_TARGET}")
  set(EIQ_NEUTRON_TARGET_DIR
      "${EIQ_NEUTRON_SDK_PATH}/target/$CACHE{EIQ_NEUTRON_TARGET}"
  )
  if(NOT EXISTS ${EIQ_NEUTRON_TARGET_DIR})
    message(
      FATAL_ERROR
        "Looking for Neutron target $CACHE{EIQ_NEUTRON_TARGET} -- not found"
    )
  else()
    message(
      STATUS
        "Looking for Neutron target $CACHE{EIQ_NEUTRON_TARGET} -- found: ${EIQ_NEUTRON_TARGET_DIR}"
    )
  endif()

  set(EIQ_NEUTRON_TARGET_INCLUDE_DIR "${EIQ_NEUTRON_TARGET_DIR}/common/include"
                                     "${EIQ_NEUTRON_TARGET_DIR}/driver/include"
  )
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

# Check if the ExecuTorch Target is aleady defined, what indicate this
# CMakeLists.txt is involved as subproject:
message(STATUS "Looking for executorch target")
if(NOT TARGET executorch)
  message(STATUS "Looking for executorch target -- not found, adding")
  set(EXECUTORCH_BUILD_NXP_NEUTRON ON)
  set(EXECUTORCH_BUILD_KERNELS_QUANTIZED ON)
  set(EXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT ON)
  add_subdirectory(
    "${EXECUTORCH_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/executorch"
    EXCLUDE_FROM_ALL
  )
else()
  message(STATUS "Looking for executorch target -- found")
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

target_compile_options(nxp_executor_runner PRIVATE -DNEUTRON_CMODEL)

target_link_libraries(
  nxp_executor_runner
  PRIVATE executorch
          executorch_kernels
          nd
          gflags::gflags
          "-Wl,--whole-archive"
          executorch_delegate_neutron
          "-Wl,--no-whole-archive"
)
