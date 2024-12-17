
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Config defining how CMake should find ExecuTorch package. CMake will search
# for this file and find ExecuTorch package if it is installed. Typical usage
# is:
#
# find_package(executorch REQUIRED)
# -------
#
# Finds the ExecuTorch library
#
# This will define the following variables:
#
#   EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
#   EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
#   EXECUTORCH_LIBRARIES    -- Libraries to link against
#
# The actual values for these variables will be different from what executorch-config.cmake
# in executorch pip package gives, but we wanted to keep the contract of exposing these
# CMake variables.

cmake_minimum_required(VERSION 3.19)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(required_lib_list executorch executorch_core portable_kernels)
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_INCLUDE_DIRS ${_root}/include ${_root}/include/executorch/runtime/core/portable_type ${_root}/lib)
foreach(lib ${required_lib_list})
  set(lib_var "LIB_${lib}")
  add_library(${lib} STATIC IMPORTED)
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}/lib"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
  target_compile_definitions(${lib} INTERFACE C10_USING_CUSTOM_GENERATED_MACROS)
  target_include_directories(${lib} INTERFACE ${_root}/include ${_root}/include/executorch/runtime/core/portable_type ${_root}/lib)
  list(APPEND EXECUTORCH_LIBRARIES ${lib})
endforeach()

# If we reach here, ET required libraries are found.
set(EXECUTORCH_FOUND ON)

target_link_libraries(executorch INTERFACE executorch_core)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(FLATCCRT_LIB flatccrt_d)
else()
  set(FLATCCRT_LIB flatccrt)
endif()

set(lib_list
    etdump
    bundled_program
    extension_data_loader
    ${FLATCCRT_LIB}
    coremldelegate
    mpsdelegate
    neuron_backend
    qnn_executorch_backend
    portable_ops_lib
    extension_module
    extension_module_static
    extension_runner_util
    extension_tensor
    extension_threadpool
    extension_training
    xnnpack_backend
    # Start XNNPACK Lib Deps
    XNNPACK
    microkernels-prod
    kleidiai
    # End XNNPACK Lib Deps
    cpuinfo
    pthreadpool
    vulkan_backend
    optimized_kernels
    cpublas
    eigen_blas
    optimized_ops_lib
    optimized_native_cpu_ops_lib
    quantized_kernels
    quantized_ops_lib
    quantized_ops_aot_lib
)
foreach(lib ${lib_list})
  # Name of the variable which stores result of the find_library search
  set(lib_var "LIB_${lib}")
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}/lib"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(NOT ${lib_var})
    message("${lib} library is not found.
            If needed rebuild with the proper options in CMakeLists.txt"
    )
  else()
    if("${lib}" STREQUAL "extension_module" AND (NOT CMAKE_TOOLCHAIN_IOS))
      add_library(${lib} SHARED IMPORTED)
    else()
      # Building a share library on iOS requires code signing, so it's easier to
      # keep all libs as static when CMAKE_TOOLCHAIN_IOS is used
      add_library(${lib} STATIC IMPORTED)
    endif()
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
    target_include_directories(${lib} INTERFACE ${_root}/include ${_root}/include/executorch/runtime/core/portable_type ${_root}/lib)
    list(APPEND EXECUTORCH_LIBRARIES ${lib})
  endif()
endforeach()
