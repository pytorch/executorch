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

cmake_minimum_required(VERSION 3.24)

include(CMakeFindDependencyMacro)
find_package(tokenizers CONFIG)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(required_lib_list executorch executorch_core portable_kernels)
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_INCLUDE_DIRS
    ${_root}/include ${_root}/include/executorch/runtime/core/portable_type/c10
    ${_root}/lib
)
foreach(lib ${required_lib_list})
  set(lib_var "LIB_${lib}")
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}/lib"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(NOT ${lib_var})
    set(EXECUTORCH_FOUND OFF)
    return()
  endif()
  list(APPEND EXECUTORCH_LIBRARIES ${lib})
endforeach()

# If we reach here, ET required libraries are found.
target_link_libraries(executorch INTERFACE executorch_core)
target_link_options_shared_lib(prim_ops_lib)

set(EXECUTORCH_FOUND ON)

target_link_libraries(executorch INTERFACE executorch_core)

# ...existing code...

# Move target_link_options_shared_lib(prim_ops_lib) to line 177 (after all target_link_libraries for executorch)
target_link_options_shared_lib(prim_ops_lib)

set(optional_lib_list
    aoti_cuda_backend
    flatccrt
    etdump
    bundled_program
    extension_data_loader
    extension_flat_tensor
    coreml_util
    coreml_inmemoryfs
    coremldelegate
    mpsdelegate
    metal_backend
    neuron_backend
    qnn_executorch_backend
    portable_ops_lib
    custom_ops
    extension_asr_runner
    extension_evalue_util
    extension_llm_runner
    extension_llm_sampler
    extension_module
    extension_module_static
    extension_runner_util
    extension_tensor
    extension_threadpool
    extension_training
    xnnpack_backend
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
    openvino_backend
    torchao_ops_executorch
    torchao_kernels_aarch64
)

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

# target_link_options_shared_lib(prim_ops_lib) is now called automatically in executorch-config.cmake
target_link_options_shared_lib(prim_ops_lib)

set(shared_lib_list
  # executorch -- size tests fail due to regression if we include this and I'm not sure it's needed.
  optimized_native_cpu_ops_lib
  portable_ops_lib
  quantized_ops_lib
  xnnpack_backend
  vulkan_backend
  quantized_ops_aot_lib)
foreach(lib ${shared_lib_list})
  if(TARGET ${lib})
    list(APPEND EXECUTORCH_LIBRARIES ${lib})
  else()
    message("${lib} library is not found.
             If needed rebuild with the proper options in CMakeLists.txt"
    )
  endif()
endforeach()

# The ARM baremetal size test's CMAKE_TOOLCHAIN_FILE apparently doesn't prevent
# our attempts to find_library(dl) from succeeding when building ExecuTorch, but
# that call finds the host system's libdl and there is no actual libdl available
# when building for the actual final baremetal.
get_property(
  FIXED_EXECUTORCH_CORE_LINK_LIBRARIES
  TARGET executorch_core
  PROPERTY INTERFACE_LINK_LIBRARIES
)
list(REMOVE_ITEM FIXED_EXECUTORCH_CORE_LINK_LIBRARIES $<LINK_ONLY:dl>)
set_property(
  TARGET executorch_core PROPERTY INTERFACE_LINK_LIBRARIES
                                  ${FIXED_EXECUTORCH_CORE_LINK_LIBRARIES}
)
