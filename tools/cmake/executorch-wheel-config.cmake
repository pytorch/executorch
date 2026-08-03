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
# Component targets are defined only when the wheel ships that component, so the
# set depends on which wheel is installed. Each one carries the runtime
# dependency and, for a registration-only library, the link options that keep it
# from being dropped. The names, when present, are:
#
# executorch::threadpool executorch::kernels executorch::xnnpack_backend
# executorch::cuda_backend
#
# Check with if(TARGET executorch::<name>) rather than assuming one exists.
#
# 3.28 rather than something older: the imported targets below export
# "$ORIGIN"-relative runtime paths as link options, and CMake writes that token
# incorrectly before 3.28. Versions 3.24 through 3.27 emit a doubled dollar with
# the Makefile generator and a bare dollar with Ninja, so a consumer builds and
# runs in place, because the absolute package directory is also recorded, then
# fails once it is deployed somewhere else.
cmake_minimum_required(VERSION 3.28)

# Everything is resolved relative to this file so the wheel stays relocatable:
# no absolute path from the machine that built it is baked in here. The file is
# installed both under share/cmake, which the historical contract uses, and
# under lib/cmake/executorch, which a plain CMAKE_PREFIX_PATH pointed at the
# package root can discover, so the root is located by a marker rather than a
# fixed depth.
find_path(
  _executorch_package_root include/executorch
  PATHS "${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_LIST_DIR}/../.."
        "${CMAKE_CURRENT_LIST_DIR}/../../.."
  NO_DEFAULT_PATH
  # NO_CACHE so the search runs on every configure. A cached result would survive
  # the package being upgraded or relocated in place and keep naming a directory
  # that has moved, which is worse than reporting it as not found.
  NO_CACHE
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

# Define an imported target for one shipped component library.
#
# A component is a prebuilt shared library next to the runtime, such as the CPU
# kernels or a delegate backend. Without a target for each one, a consumer has
# to find the file itself and decide how to keep it on the link line, which
# means depending on the wheel's private layout. The retention part matters
# most: a registration-only library has no symbol the application references, so
# a normal link drops it and its registration never runs.
#
# Call as: executorch_define_component(<target suffix> <library base name>)
function(executorch_define_component _suffix _library_name)
  file(GLOB _candidates
       "${_executorch_package_root}/lib/lib${_library_name}.so"
       "${_executorch_package_root}/lib/lib${_library_name}.so.*"
  )
  if(NOT _candidates)
    return()
  endif()
  # An unversioned name sorts first, so a development symlink wins over the
  # versioned file when both are present.
  list(SORT _candidates)
  list(GET _candidates 0 _library)

  set(_target "executorch::${_suffix}")
  if(NOT TARGET ${_target})
    add_library(${_target} SHARED IMPORTED)
  endif()
  set_target_properties(
    ${_target}
    PROPERTIES IMPORTED_LOCATION "${_library}"
               INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
               INTERFACE_COMPILE_FEATURES cxx_std_17
               INTERFACE_COMPILE_DEFINITIONS C10_USING_CUSTOM_GENERATED_MACROS
  )
  # Every component resolves the runtime from the same shared library, so record
  # that rather than leaving a consumer to link both by hand.
  if(TARGET executorch::runtime)
    set_property(
      TARGET ${_target}
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES executorch::runtime
    )
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(
      TARGET ${_target}
      APPEND
      PROPERTY INTERFACE_LINK_OPTIONS
               "LINKER:-rpath,$ORIGIN"
               "LINKER:-rpath,$ORIGIN/../lib"
               # One option per component rather than a shared push-state pair:
               # CMake removes duplicate link options, so repeating the same
               # push-state text for a second component silently drops its
               # scoping and the library goes back to being --as-needed. Naming
               # the library inside the same option keeps each one distinct.
               #
               # --no-as-needed applies only to what follows within the pushed
               # state, so the pop restores whatever the consumer had.
               "LINKER:--push-state,--no-as-needed,${_library},--pop-state"
    )
  elseif(APPLE)
    set_property(
      TARGET ${_target}
      APPEND
      PROPERTY INTERFACE_LINK_OPTIONS "SHELL:-force_load ${_library}"
    )
  endif()
  set(EXECUTORCH_LIBRARIES
      ${EXECUTORCH_LIBRARIES} ${_target}
      PARENT_SCOPE
  )
endfunction()

executorch_define_component(threadpool executorch_threadpool)

executorch_define_component(kernels executorch_optimized_native_cpu_ops_lib)

executorch_define_component(xnnpack_backend executorch_xnnpack_backend)

executorch_define_component(cuda_backend executorch_cuda_backend)

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
  # Also clear any normal-scope value, since find_library writes the cache entry
  # while a plain variable of the same name can shadow it and still look like a
  # successful discovery.
  unset(_portable_lib_LIBRARY)
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
    # SHARED, not STATIC: this resolves to the Python extension module, which is
    # a shared object. Declaring it static makes CMake treat it as an archive,
    # which changes how it is placed on a link line and how runtime paths are
    # handled.
    add_library(_portable_lib SHARED IMPORTED)
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
               # The same definition the runtime target carries. A custom-op
               # build that links only this target still compiles against the
               # same headers and needs it too.
               INTERFACE_COMPILE_DEFINITIONS C10_USING_CUSTOM_GENERATED_MACROS
  )
endif()

# find_package checks <package name>_FOUND, which is case-sensitive and does not
# match the EXECUTORCH_FOUND spelling this file documents. Without this, a
# REQUIRED find_package would succeed even when nothing usable was located.
set(executorch_FOUND ${EXECUTORCH_FOUND})
if(NOT executorch_FOUND AND executorch_FIND_REQUIRED)
  message(
    FATAL_ERROR
      "Found the ExecuTorch package but neither the shared runtime nor the Python "
      "extension could be located inside it."
  )
endif()

# Component requests are answered from the targets that were actually defined above,
# so a consumer asking for a component this wheel does not ship gets told at
# configure time rather than at link or load time. Without this a REQUIRED request
# for a missing component, or for a name that does not exist at all, would configure
# and then fail much later.
#
# The check is written out rather than using check_required_components, which comes
# from a module a package config cannot assume is already included.
foreach(_component ${executorch_FIND_COMPONENTS})
  if(TARGET executorch::${_component})
    set(executorch_${_component}_FOUND TRUE)
  else()
    set(executorch_${_component}_FOUND FALSE)
    if(executorch_FIND_REQUIRED_${_component})
      set(executorch_FOUND FALSE)
      set(executorch_NOT_FOUND_MESSAGE
          "this ExecuTorch package does not provide the required component '${_component}'"
      )
    endif()
  endif()
endforeach()
if(NOT executorch_FOUND AND executorch_FIND_REQUIRED)
  message(FATAL_ERROR "${executorch_NOT_FOUND_MESSAGE}")
endif()
