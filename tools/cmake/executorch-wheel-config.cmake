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
# In addition to the legacy variables above, this config defines namespaced
# imported targets for the prebuilt delegate-only C++ SDK when the corresponding
# static libraries are shipped in the wheel (see the "C++ SDK targets" section
# below):
#
# executorch::core                 -- executorch_core (runtime, no ops)
# executorch::runtime              -- executorch (adds primitive ops)
# executorch::extension_data_loader executorch::extension_flat_tensor
# executorch::extension_named_data_map executorch::extension_tensor
# executorch::extension_module
#
cmake_minimum_required(VERSION 3.19)

# ---------------------------------------------------------------------------
# Legacy: discover the CPython _portable_lib extension for custom-op authors.
# This keeps `find_package(executorch)` working for prebuilt custom-op
# extensions that link the Python runtime module, unchanged from before.
# ---------------------------------------------------------------------------

# Find python
if(DEFINED ENV{CONDA_DEFAULT_ENV} AND NOT $ENV{CONDA_DEFAULT_ENV} STREQUAL
                                      "base"
)
  set(PYTHON_EXECUTABLE python)
else()
  set(PYTHON_EXECUTABLE python3)
endif()

# Get the Python version and platform information
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE EXT_SUFFIX
  RESULT_VARIABLE SYSCONFIG_RESULT
  ERROR_VARIABLE SYSCONFIG_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(EXECUTORCH_INCLUDE_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../../include"
    "${CMAKE_CURRENT_LIST_DIR}/../../include/executorch/runtime/core/portable_type/c10"
)
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_FOUND OFF)

# Only discover the portable Python module when we could read EXT_SUFFIX;
# probing with an empty suffix would match a wrong/generic file. A missing
# suffix is not fatal because a pure-C++ consumer (see the C++ SDK section
# below) does not need the Python extension at all.
if(SYSCONFIG_RESULT EQUAL 0)
  message(STATUS "Sysconfig extension suffix: ${EXT_SUFFIX}")
  find_library(
    _portable_lib_LIBRARY
    NAMES _portable_lib${EXT_SUFFIX}
    PATHS "${CMAKE_CURRENT_LIST_DIR}/../../extension/pybindings/"
  )
else()
  message(
    WARNING
      "Failed to retrieve sysconfig config var EXT_SUFFIX: ${SYSCONFIG_ERROR}. "
      "The _portable_lib Python runtime target will not be available; the C++ "
      "SDK targets (executorch::*) are unaffected."
  )
endif()

if(_portable_lib_LIBRARY)
  set(EXECUTORCH_FOUND ON)
  message(
    STATUS "ExecuTorch portable library is found at ${_portable_lib_LIBRARY}"
  )
  list(APPEND EXECUTORCH_LIBRARIES _portable_lib)
  if(NOT TARGET _portable_lib)
    add_library(_portable_lib STATIC IMPORTED)
    # PyTorch requires C++20, so pybindings must be compiled with C++20.
    set_target_properties(
      _portable_lib
      PROPERTIES IMPORTED_LOCATION "${_portable_lib_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
                 CXX_STANDARD 20
    )
  endif()
endif()

# ---------------------------------------------------------------------------
# C++ SDK targets (delegate-only). Defined only when the prebuilt static
# archives are present in the wheel (they are shipped alongside this config
# under ../../lib). This lets a C++ application link the ExecuTorch runtime and
# the common runtime extensions without an ExecuTorch source checkout:
#
# find_package(executorch REQUIRED) target_link_libraries(app PRIVATE
# executorch::runtime executorch::extension_module executorch::extension_tensor)
#
# The set is intentionally libtorch-free and excludes CPU operator/kernel
# libraries; delegates (e.g. TensorRT, CUDA) supply their own compute. If your
# model needs portable CPU operators, link a kernel library in addition.
# ---------------------------------------------------------------------------

get_filename_component(
  _executorch_sdk_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)
set(_executorch_sdk_libdir "${_executorch_sdk_root}/lib")

# EXECUTORCH_SDK_FOUND is separate from EXECUTORCH_FOUND on purpose: the legacy
# EXECUTORCH_FOUND / EXECUTORCH_LIBRARIES contract describes the _portable_lib
# Python runtime for custom-op authors. Overloading it here would let existing
# `if(EXECUTORCH_FOUND) link(${EXECUTORCH_LIBRARIES})` code enter its branch
# with an empty library list. C++ SDK consumers should check the imported target
# (e.g. `if(TARGET executorch::runtime)`) or EXECUTORCH_SDK_FOUND.
#
# The C++ SDK ships one shared library, libexecutorch.so, which bundles the
# runtime core plus the common runtime extensions (module, tensor, data_loader,
# flat_tensor, named_data_map). Shared (not static archives) is required so that
# a separately distributed backend/delegate shared library can register into the
# one process-global registry that lives in libexecutorch.so. A backend .so is
# built "coreless" (its register_backend reference is undefined and resolves
# against libexecutorch.so at load), then force-loaded so its static-init
# registration runs.
set(EXECUTORCH_SDK_FOUND OFF)
find_library(
  _executorch_shared_LIBRARY
  NAMES executorch
  PATHS "${_executorch_sdk_libdir}"
  NO_DEFAULT_PATH
)
if(_executorch_shared_LIBRARY)
  set(EXECUTORCH_SDK_FOUND ON)
  if(NOT TARGET executorch::runtime)
    add_library(executorch::runtime SHARED IMPORTED)
    set_target_properties(
      executorch::runtime
      PROPERTIES IMPORTED_LOCATION "${_executorch_shared_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
                 INTERFACE_COMPILE_FEATURES cxx_std_17
                 INTERFACE_COMPILE_DEFINITIONS
                 "C10_USING_CUSTOM_GENERATED_MACROS"
    )
  endif()

  # Convenience aliases. libexecutorch.so already contains the core and these
  # extensions, so all names resolve to the one shared library. Provided so
  # consumer CMake can name what it uses without depending on the bundling
  # layout.
  foreach(_alias core extension_module extension_tensor extension_data_loader
                 extension_flat_tensor extension_named_data_map
  )
    if(NOT TARGET executorch::${_alias})
      add_library(executorch::${_alias} INTERFACE IMPORTED)
      set_property(
        TARGET executorch::${_alias} PROPERTY INTERFACE_LINK_LIBRARIES
                                              executorch::runtime
      )
    endif()
  endforeach()

  # CUDA delegate targets. Present only in a CUDA-enabled wheel, so each target
  # is defined only when its shared library is shipped in executorch/lib/. These
  # are real separate shared libraries (not aliases of libexecutorch.so):
  # extension_cuda holds the one process-wide caller-stream TLS, and
  # cuda_backend whole-archives the CUDA delegate so loading it registers
  # "CudaBackend" into the runtime.
  find_library(
    _executorch_extension_cuda_LIBRARY
    NAMES extension_cuda
    PATHS "${_executorch_sdk_libdir}"
    NO_DEFAULT_PATH
  )
  if(_executorch_extension_cuda_LIBRARY AND NOT TARGET
                                            executorch::extension_cuda
  )
    # caller_stream.h includes <cuda_runtime.h> and the real target links
    # CUDA::cudart PUBLIC, so reproduce that usage requirement.
    include(CMakeFindDependencyMacro)
    find_dependency(CUDAToolkit)
    add_library(executorch::extension_cuda SHARED IMPORTED)
    set_target_properties(
      executorch::extension_cuda
      PROPERTIES IMPORTED_LOCATION "${_executorch_extension_cuda_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
                 INTERFACE_COMPILE_FEATURES cxx_std_17
                 INTERFACE_LINK_LIBRARIES CUDA::cudart
    )
  endif()

  find_library(
    _executorch_cuda_backend_LIBRARY
    NAMES executorch_cuda_backend
    PATHS "${_executorch_sdk_libdir}"
    NO_DEFAULT_PATH
  )
  if(_executorch_cuda_backend_LIBRARY AND NOT TARGET executorch::cuda_backend)
    add_library(executorch::cuda_backend SHARED IMPORTED)
    set_target_properties(
      executorch::cuda_backend
      PROPERTIES IMPORTED_LOCATION "${_executorch_cuda_backend_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
                 INTERFACE_COMPILE_FEATURES cxx_std_17
    )
    set_property(
      TARGET executorch::cuda_backend
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES executorch::runtime
               executorch::extension_cuda
    )
  endif()
endif()
