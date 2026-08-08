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
# This is the wheel's own contract, written by hand rather than generated,
# because the wheel copies build products out of the build tree instead of
# running an install step.
#
# It is NOT identical to the in-tree package config. That one exposes the
# build's own bare target names, such as executorch and xnnpack_backend, while
# this one exposes namespaced imported targets like executorch::runtime, because
# a wheel consumer links prebuilt files rather than participating in the build.
# Consumer code written against one therefore does not configure against the
# other. The end state that removes the difference is a staged install whose
# generated targets file the wheel ships, so the source and wheel contracts
# become the same object rather than two things that must agree.
# -------
#
# Finds the ExecuTorch library
#
# This will define the following variables:
#
# EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
# EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
# EXECUTORCH_LIBRARIES    -- Libraries to link against. Includes the prebuilt
# Python extension, whose Python symbols resolve inside an interpreter, so link
# the imported targets below instead when building a standalone application.
# EXECUTORCH_BUILD_VERSION -- The full version this package was built from,
# including any prerelease suffix and local version label. Compare this when an
# exact build pairing is required, since the CMake package version keeps only
# the numeric part.
#
# and, when the prebuilt shared runtime is present, the imported target:
#
# executorch::runtime     -- The prebuilt C++ runtime (libexecutorch.so). Loads
# and executes a program, and deliberately carries no operator kernels, so
# running a model needs a kernel component as well.
#
# Component targets are defined only when the wheel ships that component, so the
# set depends on which wheel is installed. Each one carries the runtime
# dependency and, for a registration-only library, the link options that keep it
# from being dropped. The names, when present, are:
#
# executorch::kernels_optimized -- The CPU operator kernels. Needed to run a
# model. executorch::backend_xnnpack   -- The XNNPACK delegate.
# executorch::threadpool        -- The shared thread pool. executorch::etdump --
# The profiler. Links only: its concrete profiler is declared in a header that
# cannot ship, because that header reaches one generated inside a PyTorch
# installation and this package deliberately does not require PyTorch. Useful to
# a build that already has the headers from a source checkout, and it keeps
# every consumer on one copy of the profiler.
#
# Check with if(TARGET executorch::<name>) rather than assuming one exists. A
# namespaced name that was never defined is a configure-time error that names
# the component, so a consumer who links one unconditionally gets a clear
# failure rather than a broken build. Guarding is still worth doing, because a
# component's absence is a legitimate state: a CPU-only wheel ships no
# accelerator delegate, and a consumer that guards adapts instead of failing.
#
# The floor stays where it was, so a consumer that only wants the long-standing
# variables and the prebuilt Python extension keeps working on the CMake it
# already has. The shared-runtime targets below need more than this and check
# for it themselves.
cmake_minimum_required(VERSION 3.19)

# The imported targets below export "$ORIGIN"-relative runtime paths as link
# options, and CMake writes that token incorrectly before 3.28. Versions 3.24
# through 3.27 emit a doubled dollar with the Makefile generator and a bare
# dollar with Ninja, so a consumer builds and runs in place, because the
# absolute package directory is also recorded, then fails once it is deployed
# somewhere else. Silently defining a target that behaves that way is worse than
# not defining it, so the targets are skipped and a consumer that asked for one
# gets a message naming the reason.
if(CMAKE_VERSION VERSION_LESS 3.28)
  set(_executorch_targets_supported FALSE)
else()
  set(_executorch_targets_supported TRUE)
endif()

# Everything is resolved relative to this file so the wheel stays relocatable:
# no absolute path from the machine that built it is baked in here. The file is
# installed both under share/cmake, which the historical contract uses, and
# under lib/cmake/executorch, which a plain CMAKE_PREFIX_PATH pointed at the
# package root can discover, so the root is located by a marker rather than a
# fixed depth.
#
# Tested directly rather than through find_path. The root is a known relative
# offset from this file, so a search adds nothing, and find_path applies the
# consumer's find-root rules: under a cross-compiling toolchain that sets
# CMAKE_FIND_ROOT_PATH_MODE_INCLUDE to ONLY it reroots these absolute paths into
# the target sysroot, finds nothing, and reports a complete package as missing.
set(_executorch_package_root "")
foreach(_candidate
        "${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_LIST_DIR}/../.."
        "${CMAKE_CURRENT_LIST_DIR}/../../.."
)
  # share/cmake identifies the package root and only exists there. A generic
  # marker such as include/executorch can also appear one level down, in which
  # case the search from lib/cmake/executorch would stop at lib/ and resolve the
  # wrong root.
  if(EXISTS "${_candidate}/share/cmake/executorch-config.cmake")
    set(_executorch_package_root "${_candidate}")
    break()
  endif()
endforeach()

# Normalise the result before it is used to build paths. The search can return a
# directory with a trailing separator, which then appears doubled in every path
# derived from it and in the message reporting where the runtime was found.
if(_executorch_package_root)
  string(REGEX REPLACE "/+$" "" _executorch_package_root
                       "${_executorch_package_root}"
  )
endif()

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
# The full version this package was built from. It lives in the generated
# version file, which CMake includes in a throwaway scope while deciding whether
# the package is acceptable, so nothing assigned there reaches a consumer.
# Reading that file here, from the config, is what makes the value visible. Both
# files are installed side by side, so the path is fixed relative to this one.
set(_executorch_version_file
    "${CMAKE_CURRENT_LIST_DIR}/executorch-config-version.cmake"
)
if(EXISTS "${_executorch_version_file}")
  file(STRINGS "${_executorch_version_file}" _executorch_version_lines
       REGEX "^set\\(EXECUTORCH_BUILD_VERSION"
  )
  foreach(_line IN LISTS _executorch_version_lines)
    if(_line MATCHES "\"([^\"]+)\"")
      set(EXECUTORCH_BUILD_VERSION "${CMAKE_MATCH_1}")
    endif()
  endforeach()
endif()
unset(_executorch_version_file)

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
# This depends on an invariant on the build side: every library the wheel ships
# carries a VERSION and SOVERSION, so its file name ends in a major. A library
# built without them ships as a bare .so, and while the glob below still finds
# it, nothing then pins the major a consumer linked against, which is the
# guarantee the SONAME exists to provide.
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
if(_executorch_runtime_library AND NOT _executorch_targets_supported)
  message(
    STATUS
      "executorch: the prebuilt runtime is present but its imported targets need CMake 3.28 or "
      "newer, because older versions write the \$ORIGIN token in a runtime search path "
      "incorrectly. The long-standing EXECUTORCH_LIBRARIES and the prebuilt Python extension are "
      "unaffected."
  )
elseif(_executorch_runtime_library)
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
    # This file ran already in the same configure, because another subproject
    # also called find_package. Redefining the target would be an error, so keep
    # the one that is already there.
    message(
      STATUS "executorch: executorch::runtime is already defined, reusing it"
    )
  else()
    add_library(executorch::runtime SHARED IMPORTED)
    set_target_properties(
      executorch::runtime
      PROPERTIES IMPORTED_LOCATION "${_executorch_runtime_library}"
                 INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
                 INTERFACE_COMPILE_FEATURES cxx_std_17
                 INTERFACE_COMPILE_DEFINITIONS
                 C10_USING_CUSTOM_GENERATED_MACROS
    )
    # Consumers get the wheel's lib/ directory in their RUNPATH automatically,
    # because CMake adds the imported library's directory. Also record
    # $ORIGIN-relative entries so an application deployed next to a copy of the
    # runtime keeps working without relinking or LD_LIBRARY_PATH. $ORIGIN is a
    # loader token, so it belongs only in RUNPATH, never in IMPORTED_LOCATION.
    #
    # $ORIGIN is named before the wheel's own directory. An application deployed
    # beside a copy of the runtime has to find that copy, and the loader takes
    # the first match, so putting the install directory first would keep sending
    # a relocated application back to the original wheel for as long as it
    # remains installed. That also makes a relocation test that deletes the
    # original pass for the wrong reason.
    #
    # The cost, measured rather than assumed: a library that merely shares this
    # SONAME and sits in the application's own directory will win. That is what
    # $ORIGIN means in every package that uses it, and a package cannot offer
    # relocation while also refusing to honour what the user placed beside their
    # binary. The consequence worth worrying about, a delegate pairing with a
    # different registry, is caught directly by the single-registry checks,
    # which inspect what the shipped libraries define instead of trusting the
    # loader's choice.
    #
    # The absolute entry cannot simply be dropped to avoid the question: an
    # application built against the installed package fails to start without it.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      # $ORIGIN-relative entries so a relocated application keeps working. The
      # package's own directory does not need to be added here: CMake already
      # puts it in the consumer's runtime search path because the imported
      # library is named by absolute path, which is also what makes an
      # application built against the installed package start at all.
      set_property(
        TARGET executorch::runtime
        APPEND
        PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,$ORIGIN"
                 "LINKER:-rpath,$ORIGIN/../lib"
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
  # Same reason the runtime target is skipped on older CMake: a component target
  # exports an $ORIGIN-relative search path, and a version that writes it wrong
  # produces a target that works in place and fails once deployed.
  if(NOT _executorch_targets_supported)
    return()
  endif()
  _executorch_find_library(_library "lib${_library_name}")
  if(NOT _library)
    return()
  endif()

  set(_target "executorch::${_suffix}")
  if(TARGET ${_target})
    # This file ran already in the same configure, because another subproject
    # also called find_package. Redefining the target would be an error, so keep
    # the one that is already there.
    message(STATUS "executorch: ${_target} is already defined, reusing it")
    # Still advertise it. This file runs again whenever another subproject calls
    # find_package, and that run starts from an empty EXECUTORCH_LIBRARIES, so
    # returning here would hand the second caller a list with the runtime but
    # none of the components. A consumer linking that variable would then be
    # missing its kernels and fail at load with an unregistered operator.
    set(EXECUTORCH_LIBRARIES
        ${EXECUTORCH_LIBRARIES} ${_target}
        PARENT_SCOPE
    )
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
  # Guarded on Linux because these are GNU linker options. A wheel only ships
  # these components on Linux, so a consumer configured for another system is
  # either cross-compiling from the wrong package or has nothing to retain.
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
  endif()
  set(EXECUTORCH_LIBRARIES
      ${EXECUTORCH_LIBRARIES} ${_target}
      PARENT_SCOPE
  )
endfunction()

executorch_define_component(threadpool executorch_threadpool)

# The merged CPU kernels. Documented as a component and asserted by the release
# checks, so it has to be defined here or a consumer following the documentation
# gets a bare name that CMake hands to the linker as a literal flag.
executorch_define_component(kernels_optimized executorch_kernels_optimized)
# The quantized kernels, optional in the same way: a wheel built without them
# simply has no such library and the component is not defined.
executorch_define_component(kernels_quantized executorch_kernels_quantized)
# The profiler. A C++ application could not record timing data from an installed
# package before, because the implementation shipped only inside the Python
# extension.
executorch_define_component(etdump executorch_etdump)

# The switch a source build sets, on the runtime rather than on the thread pool
# target. The guarded declaration lives in a runtime header that every component
# exposes, and it selects between an extern declaration and a local inline
# definition. Putting it on the thread pool alone means a consumer with one
# translation unit linking that component and another linking only the kernels
# compiles two different definitions of the same function into one program, and
# the serial one silently wins wherever it was inlined.
#
# Unconditional because the wheel always ships the thread pool alongside the
# runtime, so there is no shipped configuration where the serial fallback is
# correct.
if(TARGET executorch::runtime)
  set_property(
    TARGET executorch::runtime
    APPEND
    PROPERTY INTERFACE_COMPILE_DEFINITIONS ET_USE_THREADPOOL
  )
  # The definition selects a declaration, and the thread pool library holds the
  # only definition of what it declares, so a consumer linking just the runtime
  # would fail to link. Carried on the runtime rather than left to the caller,
  # since the caller cannot see which header a compile definition on an imported
  # target switched.
  if(TARGET executorch::threadpool)
    set_property(
      TARGET executorch::runtime
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES executorch::threadpool
    )
  endif()
endif()

executorch_define_component(backend_xnnpack executorch_backend_xnnpack)

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
  set(_portable_lib_LIBRARY "")
else()
  message(
    FATAL_ERROR
      "Failed to retrieve sysconfig config var EXT_SUFFIX: ${SYSCONFIG_ERROR}"
  )
endif()

if(EXT_SUFFIX)
  # Tested directly rather than through find_library, for the same reason as the
  # package root: the path and the file name are both already known, so a search
  # only adds the consumer's find-root rules, which reroot an absolute wheel
  # path into a cross-compile sysroot and report a present extension as missing.
  set(_portable_lib_candidate
      "${_executorch_package_root}/extension/pybindings/_portable_lib${EXT_SUFFIX}"
  )
  if(EXISTS "${_portable_lib_candidate}")
    set(_portable_lib_LIBRARY "${_portable_lib_candidate}")
  else()
    set(_portable_lib_LIBRARY "")
  endif()
endif()

if(_portable_lib_LIBRARY)
  set(EXECUTORCH_FOUND ON)
  message(
    STATUS "ExecuTorch portable library is found at ${_portable_lib_LIBRARY}"
  )
  list(APPEND EXECUTORCH_LIBRARIES _portable_lib)
  if(TARGET _portable_lib)
    # This file ran already in the same configure, because another subproject
    # called find_package too. No in-tree target uses this name, so it can only
    # be the imported one defined below, and re-setting its properties to the
    # same values is harmless.
    message(STATUS "executorch: _portable_lib is already defined, reusing it")
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
  # The extension links the runtime rather than containing it, so it no longer
  # satisfies the runtime symbols a custom-op library references. Put the
  # shipped runtime on this target's interface, which is where the definitions
  # moved to, so an out-of-tree operator project keeps building and loading
  # against the extension exactly as it did before. Without this a custom
  # operator links and then fails to load with an undefined runtime symbol.
  #
  # The file path rather than executorch::runtime, because that target is only
  # defined on CMake 3.28 or newer while this one has no such requirement.
  if(_executorch_runtime_library)
    set_property(
      TARGET _portable_lib
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES "${_executorch_runtime_library}"
    )
  endif()
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
      # Naming the CMake version when that is the cause saves a consumer from
      # concluding the component is missing from the package, which is the wrong
      # thing to go looking for.
      if(NOT _executorch_targets_supported)
        # One string rather than several arguments. Several make a list, and
        # message() joins a list with semicolons, which lands separators mid
        # sentence.
        string(
          CONCAT
            executorch_NOT_FOUND_MESSAGE
            "the required component '${_component}' needs CMake 3.28 or newer, because older "
            "versions write the \$ORIGIN token in a runtime search path incorrectly; this "
            "package is otherwise usable through EXECUTORCH_LIBRARIES"
        )
      else()
        # One string rather than several arguments. Several make a list, and
        # message() joins a list with semicolons, which lands separators mid
        # sentence.
        string(
          CONCAT
            executorch_NOT_FOUND_MESSAGE
            "this ExecuTorch package does not provide the required component '${_component}'"
        )
      endif()
    endif()
  endif()
endforeach()
if(NOT executorch_FOUND AND executorch_FIND_REQUIRED)
  message(FATAL_ERROR "${executorch_NOT_FOUND_MESSAGE}")
endif()
