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
#
# This file describes the same contract as the in-tree package config, but is written by hand
# rather than generated, because the wheel copies build products out of the build tree instead
# of running an install step. That leaves two descriptions of one contract, which is why a
# target already defined by an in-tree build has to be detected and left alone below.
#
# The end state that removes the duplication is a staged install whose generated targets file
# the wheel ships, so the source and wheel contracts become the same object rather than two
# things that must agree.
# -------
#
# Finds the ExecuTorch library
#
# This will define the following variables:
#
# EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
# EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
# EXECUTORCH_LIBRARIES    -- Libraries to link against
# EXECUTORCH_BUILD_VERSION -- The full version this package was built from, including
#                            any prerelease suffix and local version label. Compare this
#                            when an exact build pairing is required, since the CMake
#                            package version keeps only the numeric part.
# EXECUTORCH_BUILD_VERSION -- The full version this package was built from, including
#                            any prerelease suffix and local version label. Compare this
#                            when an exact build pairing is required, since the CMake
#                            package version keeps only the numeric part.
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
  _executorch_package_root
  # share/cmake identifies the package root and only exists there. A generic
  # marker such as include/executorch can also appear one level down, in which
  # case the search from lib/cmake/executorch would stop at lib/ and resolve the
  # wrong root.
  NAMES share/cmake/executorch-config.cmake
  PATHS "${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_LIST_DIR}/../.."
        "${CMAKE_CURRENT_LIST_DIR}/../../.."
  NO_DEFAULT_PATH
  # NO_CACHE so the search runs on every configure. A cached result would
  # survive the package being upgraded or relocated in place and keep naming a
  # directory that has moved, which is worse than reporting it as not found.
  NO_CACHE
)

# Both directories are needed for a usable package. The C10 compatibility
# headers are not optional: core headers such as runtime/core/array_ref.h
# include c10 unconditionally, so a package missing them cannot compile anything
# that touches the runtime API.
#
# A missing directory is reported as not-found rather than raised here, so an
# optional find_package gets a FALSE answer instead of a dead build. The
# REQUIRED handling at the bottom of this file turns it into an error when the
# caller asked for one.
set(_executorch_c10_include
    "${_executorch_package_root}/include/executorch/runtime/core/portable_type/c10"
)
set(EXECUTORCH_INCLUDE_DIRS "${_executorch_package_root}/include"
                            "${_executorch_c10_include}"
)
foreach(_required_include ${EXECUTORCH_INCLUDE_DIRS})
  if(NOT EXISTS "${_required_include}")
    message(
      STATUS "ExecuTorch package at ${_executorch_package_root} is missing "
             "${_required_include}, so nothing can compile against it."
    )
    set(EXECUTORCH_INCLUDE_DIRS)
    set(EXECUTORCH_LIBRARIES)
    set(EXECUTORCH_FOUND OFF)
    set(executorch_FOUND FALSE)
    return()
  endif()
endforeach()

set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_FOUND OFF)

# Locate one shipped library by base name.
#
# A wheel ships a single file per library, named for its SONAME, so the major is
# read from the shipped names rather than hardcoded here. Sets <output> to the
# full path, or to an empty string when the wheel does not carry that library.
#
# This depends on an invariant on the build side: every library the wheel ships carries a
# VERSION and SOVERSION, so its file name ends in a major. A library built without them ships
# as a bare .so, and while the glob below still finds it, nothing then pins the major a
# consumer linked against, which is the guarantee the SONAME exists to provide.
function(_executorch_find_library _output _base_name)
  set(${_output}
      ""
      PARENT_SCOPE
  )
  file(GLOB _matches "${_executorch_package_root}/lib/${_base_name}.so"
       "${_executorch_package_root}/lib/${_base_name}.so.*"
  )
  list(LENGTH _matches _count)
  if(_count EQUAL 0)
    return()
  endif()
  # Highest major wins, so a package that somehow carries two does not silently
  # select by string order. Natural ordering keeps .2 below .10.
  list(
    SORT _matches
    COMPARE NATURAL
    ORDER DESCENDING
  )
  list(GET _matches 0 _selected)
  set(${_output}
      "${_selected}"
      PARENT_SCOPE
  )
endfunction()

# The prebuilt runtime.
_executorch_find_library(_executorch_runtime_library libexecutorch)
if(_executorch_runtime_library)
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
if(TARGET executorch::runtime)
    # An in-tree build defines this name, sometimes as an ALIAS whose properties cannot be
    # set. A consumer that both adds this project as a subdirectory and calls find_package
    # should keep the target it is already building, so skip the whole definition.
    message(STATUS "executorch: executorch::runtime is already defined, leaving it as is")
  else()
    add_library(executorch::runtime SHARED IMPORTED)
    set_target_properties(
      executorch::runtime
      PROPERTIES IMPORTED_LOCATION "${_executorch_runtime_library}"
                 INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
                 INTERFACE_COMPILE_FEATURES cxx_std_17
                 INTERFACE_COMPILE_DEFINITIONS C10_USING_CUSTOM_GENERATED_MACROS
    )
    # Consumers get the wheel's lib/ directory in their RUNPATH automatically, because
    # CMake adds the imported library's directory. Also record $ORIGIN-relative entries so
    # an application deployed next to a copy of the runtime keeps working without relinking
    # or LD_LIBRARY_PATH. $ORIGIN is a loader token, so it belongs only in RUNPATH, never
    # in IMPORTED_LOCATION.
    #
    # $ORIGIN is named before the wheel's own directory. An application deployed beside a copy
    # of the runtime has to find that copy, and the loader takes the first match, so putting
    # the install directory first would keep sending a relocated application back to the
    # original wheel for as long as it remains installed. That also makes a relocation test
    # that deletes the original pass for the wrong reason.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      get_filename_component(
        _executorch_runtime_dir "${_executorch_runtime_library}" DIRECTORY
      )
      set_property(
        TARGET executorch::runtime
        APPEND
        PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,$ORIGIN"
                 "LINKER:-rpath,$ORIGIN/../lib"
                 "LINKER:-rpath,${_executorch_runtime_dir}"
      )
    endif()
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
  _executorch_find_library(_library "lib${_library_name}")
  if(NOT _library)
    return()
  endif()

  set(_target "executorch::${_suffix}")
  if(TARGET ${_target})
    # An in-tree build defines these names, sometimes as an ALIAS whose properties
    # cannot be set. A consumer that both adds this project as a subdirectory and
    # calls find_package should keep the target it is already building.
    message(STATUS "executorch: ${_target} is already defined, leaving it as is")
    return()
  endif()
  add_library(${_target} SHARED IMPORTED)
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
      # One LINKER: option with a comma rather than a SHELL: string with a
      # space: SHELL splits on spaces, so a library path containing one would
      # reach the linker as two broken arguments.
      PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-force_load,${_library}"
    )
  endif()
  set(EXECUTORCH_LIBRARIES
      ${EXECUTORCH_LIBRARIES} ${_target}
      PARENT_SCOPE
  )
endfunction()

executorch_define_component(threadpool executorch_threadpool)


executorch_define_component(xnnpack_backend executorch_xnnpack_backend)

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
    # NO_CACHE for the same reason as the package root above: a cached result
    # would survive the package being upgraded or relocated in the same build
    # directory and keep naming the previous copy, or a file that no longer
    # exists.
    NO_CACHE
  )
endif()

if(_portable_lib_LIBRARY)
  set(EXECUTORCH_FOUND ON)
  message(
    STATUS "ExecuTorch portable library is found at ${_portable_lib_LIBRARY}"
  )
  list(APPEND EXECUTORCH_LIBRARIES _portable_lib)
  if(TARGET _portable_lib)
    # Already defined by an in-tree build, so keep it rather than redefining it.
    message(STATUS "executorch: _portable_lib is already defined, leaving it as is")
  else()
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

# Component requests are answered from the targets that were actually defined
# above, so a consumer asking for a component this wheel does not ship gets told
# at configure time rather than at link or load time. Without this a REQUIRED
# request for a missing component, or for a name that does not exist at all,
# would configure and then fail much later.
#
# The check is written out rather than using check_required_components, which
# comes from a module a package config cannot assume is already included.
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
