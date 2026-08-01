# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Config defining how CMake should find ExecuTorch package. CMake will search
# for this file and find ExecuTorch package if it is installed. Typical usage
# is:
#
# ~~~
# find_package(executorch REQUIRED)
# target_link_libraries(my_app PRIVATE executorch::runtime)
# ~~~
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
# and, when the prebuilt shared runtime is present, the imported target:
#
# executorch::runtime     -- The prebuilt C++ runtime (libexecutorch.so)
#
cmake_minimum_required(VERSION 3.19)

# This file is installed to <site-packages>/executorch/share/cmake, so the
# package root is two levels up. Everything is resolved relative to this file so
# the wheel stays relocatable: no absolute path from the machine that built it
# is baked in here.
get_filename_component(
  _executorch_package_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)

set(EXECUTORCH_INCLUDE_DIRS
    "${_executorch_package_root}/include"
    "${_executorch_package_root}/include/executorch/runtime/core/portable_type/c10"
)

set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_FOUND OFF)

# The prebuilt runtime. Match the versioned file rather than a hardcoded major
# so the config keeps working across releases.
file(GLOB _executorch_runtime_candidates
     "${_executorch_package_root}/lib/libexecutorch.so"
     "${_executorch_package_root}/lib/libexecutorch.so.*"
)
# An unversioned libexecutorch.so sorts before any libexecutorch.so.<major>, so
# a development symlink wins over the versioned file when both are present.
list(SORT _executorch_runtime_candidates)
list(LENGTH _executorch_runtime_candidates _executorch_runtime_count)
if(_executorch_runtime_count GREATER 0)
  list(GET _executorch_runtime_candidates 0 _executorch_runtime_library)

  set(EXECUTORCH_FOUND ON)
  message(STATUS "ExecuTorch runtime found at ${_executorch_runtime_library}")

  # The documented contract is that a consumer can link ${EXECUTORCH_LIBRARIES}.
  # Leaving it empty here would make find_package succeed while offering nothing
  # linkable to anyone who has not moved to the imported target.
  list(APPEND EXECUTORCH_LIBRARIES executorch::runtime)

  # This file can be processed more than once in a single configure, for example
  # when several subprojects each call find_package(executorch). Creating the
  # target twice is an error, so only define it once and set the properties
  # either way.
  if(NOT TARGET executorch::runtime)
    add_library(executorch::runtime SHARED IMPORTED)
  endif()
  set_target_properties(
    executorch::runtime
    PROPERTIES IMPORTED_LOCATION "${_executorch_runtime_library}"
               INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
               INTERFACE_COMPILE_FEATURES cxx_std_17
               INTERFACE_COMPILE_DEFINITIONS C10_USING_CUSTOM_GENERATED_MACROS
  )
  # Consumers get the wheel's lib/ directory in their RUNPATH automatically,
  # because CMake adds the imported library's directory. Also record
  # $ORIGIN-relative entries so an application that is deployed next to a copy
  # of the runtime keeps working without relinking or LD_LIBRARY_PATH. $ORIGIN
  # is a loader token, so it belongs only in RUNPATH, never in
  # IMPORTED_LOCATION.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(
      TARGET executorch::runtime
      APPEND
      PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,$ORIGIN"
               "LINKER:-rpath,$ORIGIN/../lib"
    )
  endif()
endif()

# Find prebuilt _portable_lib.<EXT_SUFFIX>.so. This is the legacy contract used
# to build custom-op extensions against the Python module, and is kept working
# independently of the runtime target above.

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

if(SYSCONFIG_RESULT EQUAL 0)
  message(STATUS "Sysconfig extension suffix: ${EXT_SUFFIX}")
elseif(TARGET executorch::runtime)
  # A C++ application linking only the shared runtime does not need Python at
  # all, so a missing interpreter must not fail its configure. Skip locating the
  # Python extension instead; the legacy _portable_lib target is simply not
  # offered in that case.
  message(
    STATUS
      "Python not usable, skipping the Python extension: ${SYSCONFIG_ERROR}"
  )
  set(EXT_SUFFIX "")
  # find_library caches its result, so a value left by an earlier configure
  # would survive and the extension would still be offered despite being skipped
  # here.
  unset(_portable_lib_LIBRARY CACHE)
else()
  message(
    FATAL_ERROR
      "Failed to retrieve sysconfig config var EXT_SUFFIX: ${SYSCONFIG_ERROR}"
  )
endif()

if(EXT_SUFFIX)
  find_library(
    _portable_lib_LIBRARY
    NAMES _portable_lib${EXT_SUFFIX}
    PATHS "${_executorch_package_root}/extension/pybindings/"
    # This config binds to the wheel it ships in, so a same-named library
    # elsewhere on the system must not be picked up instead.
    NO_DEFAULT_PATH
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
  endif()
  # PyTorch requires C++20, so pybindings must be compiled with C++20.
  set_target_properties(
    _portable_lib
    PROPERTIES IMPORTED_LOCATION "${_portable_lib_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
               # An interface requirement rather than CXX_STANDARD: an imported
               # target compiles nothing itself, and CXX_STANDARD does not reach
               # consumers, so a custom-op build linking this could still
               # compile
               # as C++17 and fail against headers that need C++20.
               INTERFACE_COMPILE_FEATURES cxx_std_20
  )
endif()
