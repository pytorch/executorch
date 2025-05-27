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
# EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
# EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
# EXECUTORCH_LIBRARIES    -- Libraries to link against
#
# The actual values for these variables will be different from what
# executorch-config.cmake in executorch pip package gives, but we wanted to keep
# the contract of exposing these CMake variables.

cmake_minimum_required(VERSION 3.19)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(required_lib_list executorch executorch_core portable_kernels)
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_INCLUDE_DIRS
    ${_root}/include ${_root}/include/executorch/runtime/core/portable_type/c10
    ${_root}/lib
)
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
  target_include_directories(
    ${lib}
    INTERFACE ${_root}/include
              ${_root}/include/executorch/runtime/core/portable_type/c10
              ${_root}/lib
  )
  list(APPEND EXECUTORCH_LIBRARIES ${lib})
endforeach()

# If we reach here, ET required libraries are found.
set(EXECUTORCH_FOUND ON)

target_link_libraries(executorch INTERFACE executorch_core)

set(lib_list
    flatccrt
    etdump
    bundled_program
    extension_data_loader
    extension_flat_tensor
    coreml_util
    coreml_inmemoryfs
    coremldelegate
    mpsdelegate
    neuron_backend
    qnn_executorch_backend
    portable_ops_lib
    custom_ops
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
    optimized_portable_kernels
    cpublas
    eigen_blas
    optimized_ops_lib
    optimized_native_cpu_ops_lib
    quantized_kernels
    quantized_ops_lib
    quantized_ops_aot_lib
    torchao_ops_executorch
    torchao_kernels_aarch64
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
    target_include_directories(
      ${lib}
      INTERFACE ${_root}/include
                ${_root}/include/executorch/runtime/core/portable_type/c10
                ${_root}/lib
    )
    list(APPEND EXECUTORCH_LIBRARIES ${lib})
  endif()
endforeach()

# TODO: investigate use of install(EXPORT) to cleanly handle
# target_compile_options/target_compile_definitions for everything.
if(TARGET cpublas)
  set_target_properties(
    cpublas PROPERTIES INTERFACE_LINK_LIBRARIES
                       "extension_threadpool;eigen_blas"
  )
endif()
if(TARGET optimized_kernels)
  set_target_properties(
    optimized_kernels PROPERTIES INTERFACE_LINK_LIBRARIES
                                 "executorch_core;cpublas;extension_threadpool"
  )
endif()

if(TARGET torchao_ops_executorch)
  set_target_properties(
    torchao_ops_executorch PROPERTIES INTERFACE_LINK_LIBRARIES
                                 "executorch_core;extension_threadpool;cpuinfo;pthreadpool"
  )
endif()

if(TARGET coremldelegate)
  set_target_properties(
    coremldelegate PROPERTIES INTERFACE_LINK_LIBRARIES
                              "coreml_inmemoryfs;coreml_util"
  )
endif()

if(TARGET etdump)
  set_target_properties(etdump PROPERTIES INTERFACE_LINK_LIBRARIES "flatccrt;executorch")
endif()

if(TARGET optimized_native_cpu_ops_lib)
  if(TARGET optimized_portable_kernels)
    set(_maybe_optimized_portable_kernels_lib optimized_portable_kernels)
  else()
    set(_maybe_optimized_portable_kernels_lib portable_kernels)
  endif()
  set_target_properties(
    optimized_native_cpu_ops_lib
    PROPERTIES INTERFACE_LINK_LIBRARIES
               "optimized_kernels;${_maybe_optimized_portable_kernels_lib}"
  )
endif()
if(TARGET extension_threadpool)
  target_compile_definitions(extension_threadpool INTERFACE ET_USE_THREADPOOL)
  set_target_properties(
    extension_threadpool PROPERTIES INTERFACE_LINK_LIBRARIES
                                    "cpuinfo;pthreadpool"
  )
endif()
