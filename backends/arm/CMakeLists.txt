# Copyright 2023, 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.19)
project(arm_backend)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# bare metal backend builds
if(EXECUTORCH_BUILD_ARM_BAREMETAL)

  add_compile_options("-Wall" "-Werror")

  # Third-party folder and Ethos-U driver inclued
  set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party")
  set(DRIVER_ETHOSU_INCLUDE_DIR
      "${THIRD_PARTY_ROOT}/ethos-u-core-driver/include"
  )
  include_directories(${DRIVER_ETHOSU_INCLUDE_DIR})

  set(_arm_baremetal_sources backends/arm/runtime/EthosUBackend.cpp
                             backends/arm/runtime/VelaBinStream.cpp
  )
  list(TRANSFORM _arm_baremetal_sources PREPEND "${EXECUTORCH_ROOT}/")

  add_library(executorch_delegate_ethos_u STATIC ${_arm_baremetal_sources})
  target_link_libraries(
    executorch_delegate_ethos_u PUBLIC executorch_core ethosu_core_driver
  )

  install(TARGETS executorch_delegate_ethos_u EXPORT ExecuTorchTargets)

  # end config for bare metal builds
endif()

# VGF backend builds
if(EXECUTORCH_BUILD_VGF)

  # include libvgf
  set(LIBVGF_PATH
      "${EXECUTORCH_ROOT}/examples/arm/ethos-u-scratch/ml-sdk-for-vulkan-manifest/sw/vgf-lib/"
  )

  set(VULKAN_THIRD_PARTY_PATH ${EXECUTORCH_ROOT}/backends/vulkan/third-party)
  set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
  set(VOLK_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)

  set(LIBVGF_STATIC "${LIBVGF_PATH}/build/src/libvgf.a")
  set(LIBVGF_INCLUDE "${LIBVGF_PATH}/include/")

  add_library(vgf STATIC IMPORTED)
  set_property(TARGET vgf PROPERTY IMPORTED_LOCATION "${LIBVGF_STATIC}")
  target_include_directories(vgf INTERFACE "${LIBVGF_INCLUDE}")

  # Add backend delegate for VGF
  set(_vgf_backend_sources backends/arm/runtime/VGFBackend.cpp
                           backends/arm/runtime/VGFSetup.cpp
  )

  # vgf backend
  list(TRANSFORM _vgf_backend_sources PREPEND "${EXECUTORCH_ROOT}/")
  add_library(vgf_backend ${_vgf_backend_sources})
  install(TARGETS vgf_backend EXPORT ExecuTorchTargets)
  target_include_directories(
    vgf_backend PRIVATE ${_common_include_directories} ${VULKAN_HEADERS_PATH}
                        ${VOLK_HEADERS_PATH}
  )
  target_compile_options(
    vgf_backend PRIVATE -DUSE_VULKAN_WRAPPER -DUSE_VULKAN_VOLK
  )

  target_link_libraries(vgf_backend PRIVATE executorch_core)
  target_link_libraries(vgf_backend PRIVATE vgf)
  executorch_target_link_options_shared_lib(vgf_backend)

  # end config for VGF builds
endif()
