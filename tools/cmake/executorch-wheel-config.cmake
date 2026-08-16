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
# EXECUTORCH_RUNTIME_LIBRARY_DIR -- Where the shipped libraries live. A consumer
# that installs its own binary elsewhere adds this to its INSTALL_RPATH, because
# CMake removes the entry it recorded while building.
#
cmake_minimum_required(VERSION 3.19)

# Find prebuilt _portable_lib.<EXT_SUFFIX>.so. This file should be installed
# under <site-packages>/executorch/share/cmake

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
else()
  message(
    FATAL_ERROR
      "Failed to retrieve sysconfig config var EXT_SUFFIX: ${SYSCONFIG_ERROR}"
  )
endif()

find_library(
  _portable_lib_LIBRARY
  NAMES _portable_lib${EXT_SUFFIX}
  PATHS "${CMAKE_CURRENT_LIST_DIR}/../../extension/pybindings/"
)

set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_FOUND OFF)
if(_portable_lib_LIBRARY)
  set(EXECUTORCH_FOUND ON)
  message(
    STATUS "ExecuTorch portable library is found at ${_portable_lib_LIBRARY}"
  )
  list(APPEND EXECUTORCH_LIBRARIES _portable_lib)
  add_library(_portable_lib STATIC IMPORTED)
  set(EXECUTORCH_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include)
  set_target_properties(
    _portable_lib
    PROPERTIES IMPORTED_LOCATION "${_portable_lib_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
               # PyTorch requires C++20, so anything linking this must compile
               # as
               # C++20. An interface requirement rather than CXX_STANDARD,
               # because
               # an imported target compiles nothing itself and CXX_STANDARD
               # does
               # not reach consumers, so a custom-op build could still compile
               # as
               # C++17 and fail against headers that need C++20.
               INTERFACE_COMPILE_FEATURES cxx_std_20
               # The shipped libraries are compiled with the event tracer on,
               # and
               # the profiling scope classes declare their members inside that
               # guard. Without it a consumer compiles a different, empty
               # version
               # of the same class, so its profiling scopes record nothing and
               # report no error.
               INTERFACE_COMPILE_DEFINITIONS "@EXECUTORCH_TRACER_DEFINITION@"
  )

  # The extension links the runtime rather than containing it, so it no longer
  # satisfies the runtime symbols a custom-op library references. Put the
  # shipped runtime on this target's interface, which is where the definitions
  # moved to, so an out-of-tree operator project keeps building and loading
  # against the extension exactly as it did before.
  find_library(
    EXECUTORCH_RUNTIME_LIBRARY executorch
    PATHS "${CMAKE_CURRENT_LIST_DIR}/../../lib"
    NO_DEFAULT_PATH
  )
  if(EXECUTORCH_RUNTIME_LIBRARY)
    set_property(
      TARGET _portable_lib
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES "${EXECUTORCH_RUNTIME_LIBRARY}"
    )
    # CMake adds a linked library's directory to the consumer's build tree
    # runtime search path and strips it on install, so an installed consumer
    # library reports libexecutorch.so as not found. Publish the directories
    # rather than forcing them onto the target: an interface link option reaches
    # every consumer and survives install, which would bake this machine's
    # package location into a library the consumer ships onward. A consumer that
    # installs elsewhere adds these to its own INSTALL_RPATH.
    get_filename_component(
      EXECUTORCH_RUNTIME_LIBRARY_DIR "${EXECUTORCH_RUNTIME_LIBRARY}" DIRECTORY
    )
  endif()
endif()

# find_package checks <package name>_FOUND, which is case-sensitive and does not
# match the EXECUTORCH_FOUND spelling this file documents. Without this, a
# REQUIRED find_package succeeds even when nothing usable was located, and the
# consumer goes on to link nothing.
set(executorch_FOUND ${EXECUTORCH_FOUND})
if(NOT executorch_FOUND AND executorch_FIND_REQUIRED)
  message(
    FATAL_ERROR
      "Found the ExecuTorch package but could not locate the Python extension "
      "inside it, so there is nothing to link."
  )
endif()
