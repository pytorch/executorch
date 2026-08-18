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

# Load dependencies published by enabled backend targets.
include("${CMAKE_CURRENT_LIST_DIR}/executorch-backend-dependencies.cmake"
        OPTIONAL
)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
# Included before the library list is built, because the list depends on which
# runtime this install actually contains.
include("${CMAKE_CURRENT_LIST_DIR}/ExecuTorchTargets.cmake")

# A shared build installs libexecutorch.a and libexecutorch.so side by side, and
# naming the static runtime as well as the shared one gives a consumer two
# kernel registries, which aborts during registration before main. The shared
# library carries the core, so it replaces the static runtime, but measurement
# shows it does not contain the kernels: dropping the kernels target here leaves
# a consumer with no operators at all. That target still pulls the static core
# in behind it, so a second copy remains present, and ELF resolves both to the
# executable's definition, so the program is correct.
#
# Mach-O does not: each image binds its own definition, so the copies never
# converge and code inside the shared runtime sees a different registry from the
# one the application registered into. Apple therefore keeps the all-static list
# this file handed out before the shared library existed, where there is one
# copy and nothing to diverge. Handing out the shared runtime there too means
# dropping every archive in the optional list below that is already bundled into
# it, which is a wider change than the one that added the library.
if(TARGET executorch-shared AND NOT APPLE)
  set(required_lib_list portable_kernels)
  set(EXECUTORCH_LIBRARIES executorch-shared)
else()
  set(required_lib_list executorch executorch_core portable_kernels)
  set(EXECUTORCH_LIBRARIES)
endif()
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
set(EXECUTORCH_FOUND ON)

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
    mlxdelegate
    mlx
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

foreach(lib ${optional_lib_list})
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
# when building for the actual final baremetal. Guarded because a shared install
# need not export the static core, and reading a property off a target that does
# not exist is a hard error in every consumer's find_package.
if(TARGET executorch_core)
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
endif()

# Expose MLX library and metallib path for downstream consumers
if(TARGET mlxdelegate)
  # Create imported target for mlx library if not already defined (mlx is built
  # by MLX's CMake but we need to expose it for linking)
  if(NOT TARGET mlx)
    find_library(
      _mlx_library mlx
      HINTS "${_root}/lib"
      CMAKE_FIND_ROOT_PATH_BOTH
    )
    if(_mlx_library)
      add_library(mlx STATIC IMPORTED)
      set_target_properties(mlx PROPERTIES IMPORTED_LOCATION "${_mlx_library}")
      # libmlx.a is a static archive with no transitive link deps, so re-add the
      # frameworks MLX links PUBLIC (mirrors
      # third-party/mlx/CMakeLists.txt:209). Must match the in-tree imported
      # target in backends/mlx/CMakeLists.txt.
      if(APPLE)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        find_library(QUARTZ_FRAMEWORK QuartzCore)
        if(METAL_FRAMEWORK
           AND FOUNDATION_FRAMEWORK
           AND QUARTZ_FRAMEWORK
        )
          set_target_properties(
            mlx
            PROPERTIES
              INTERFACE_LINK_LIBRARIES
              "${METAL_FRAMEWORK};${FOUNDATION_FRAMEWORK};${QUARTZ_FRAMEWORK}"
          )
        endif()
      endif()
      message(STATUS "Found mlx library at: ${_mlx_library}")
    endif()
  endif()

  # Find metallib for runtime distribution
  find_file(
    _mlx_metallib mlx.metallib
    HINTS "${_root}/lib"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(_mlx_metallib)
    set(MLX_METALLIB_PATH
        "${_mlx_metallib}"
        CACHE FILEPATH "Path to mlx.metallib for runtime distribution"
    )
    message(STATUS "Found mlx.metallib at: ${MLX_METALLIB_PATH}")
  endif()
endif()
