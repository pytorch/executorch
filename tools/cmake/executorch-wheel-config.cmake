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
# Targets here are namespaced, such as executorch::runtime, unlike the in-tree
# package config which exposes the build's own bare target names. Code written
# against one does not configure against the other.
# -------
#
# Finds the ExecuTorch library
#
# This will define the following variables:
#
# EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
#
# EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
#
# EXECUTORCH_COMPILE_DEFINITIONS -- Definitions a consumer must compile with.
# The vendored c10 headers otherwise reach for a header generated inside a
# PyTorch build, which no wheel can carry.
#
# EXECUTORCH_CXX_STANDARD -- The minimum C++ standard the shipped headers need.
# A consumer using these variables has to set it, because a compiler defaulting
# to an older standard cannot parse them.
#
# EXECUTORCH_RUNTIME_LIBRARY_DIR -- Where the shipped libraries live. A consumer
# that installs its own binary elsewhere adds this to its INSTALL_RPATH, because
# CMake removes the entry it recorded while building.
#
# EXECUTORCH_LIBRARIES    -- Libraries to link against: the prebuilt runtime and
# the components the wheel shipped, except the ones documented below as opt in.
# Not the Python extension, which carries unresolved interpreter symbols that
# only resolve inside an interpreter, so a standalone application linking it
# fails with a page of PyUnicode_InternFromString errors. A project building a
# custom operator against the extension asks for the _portable_lib target by
# name, which is the long-standing contract for that and also carries the C++20
# requirement PyTorch's headers need.
#
# EXECUTORCH_BUILD_VERSION -- The full version this package was built from,
# including any prerelease suffix and local version label. Compare this when an
# exact build pairing is required, since the CMake package version keeps only
# the numeric part.
#
# and, when the prebuilt shared runtime is present, the imported target:
#
# executorch::runtime     -- The prebuilt C++ runtime (libexecutorch.so). Loads
# and executes a program, and carries only primitive operators rather than the
# kernels a model computes with, so running a model needs a kernel component as
# well.
#
# Component targets are defined only when the wheel ships that component, so the
# set depends on which wheel is installed. Each one carries the runtime
# dependency and, for a registration-only library, the link options that keep it
# from being dropped. The names, when present, are:
#
# ~~~
# executorch::kernels_optimized  The CPU operator kernels. Needed to run a model.
# executorch::kernels_quantized  The quantized operator kernels, for a quantized
#                                model. Not part of EXECUTORCH_LIBRARIES, see
#                                below.
# executorch::backend_xnnpack    The XNNPACK delegate.
# executorch::backend_cuda       The CUDA delegate. Linux only.
# executorch::extension_cuda     The CUDA stream extension. Linux only.
# executorch::backend_openvino   The OpenVINO delegate. Linux only. Opens the
#                                OpenVINO runtime by name, which a C++ program
#                                installs and points OPENVINO_LIB_PATH at.
# executorch::threadpool         The shared thread pool.
# executorch::etdump             The profiler.
# ~~~
#
# EXECUTORCH_LIBRARIES carries every component except the quantized kernels,
# which a consumer names explicitly instead. The export-time plugin that
# executorch.kernels.quantized loads carries its own copy of those kernels, so a
# process holding both stops on a repeated operator registration, and a consumer
# linking the aggregate would inherit that without asking for it.
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
# options, and CMake writes that token incorrectly before 3.28. Measured rather
# than inferred, on a consumer linking such a target:
#
# ~~~
#   3.24.3, 3.27.9   Makefiles double the dollar sign, Ninja drops the name
#   3.28.4, 3.31.8   both write the token correctly
# ~~~
#
# Either broken form leaves a consumer building and running in place, because
# the absolute package directory is also recorded, then failing once it is
# deployed somewhere else. Silently defining a target that behaves that way is
# worse than not defining it, so the targets are skipped and a consumer that
# asked for one gets a message naming the reason.
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
# The same definitions the imported targets carry. A consumer on CMake older
# than 3.28 gets no imported targets and links through EXECUTORCH_LIBRARIES
# instead, and without the first of these it could not compile at all: the
# vendored c10 headers reach for a header generated inside a PyTorch build that
# no wheel can carry. The tracer switch is here because the shipped libraries
# are compiled with it, and the profiling scope classes declare their members
# inside that guard, so a consumer compiling without it sees a smaller, empty
# version of the same class and its profiling scopes record nothing while
# reporting no error.
set(EXECUTORCH_COMPILE_DEFINITIONS C10_USING_CUSTOM_GENERATED_MACROS
                                   "@EXECUTORCH_TRACER_DEFINITION@"
)
# The standard the imported targets require as a compile feature. Exported as a
# variable too, because a consumer on CMake older than 3.28 gets no imported
# targets and would otherwise compile these headers with whatever its compiler
# defaults to.
set(EXECUTORCH_CXX_STANDARD 17)
foreach(_required_include ${EXECUTORCH_INCLUDE_DIRS})
  if(NOT EXISTS "${_required_include}")
    message(
      STATUS "ExecuTorch package at ${_executorch_package_root} is missing "
             "${_required_include}, so nothing can compile against it."
    )
    set(EXECUTORCH_INCLUDE_DIRS)
    set(EXECUTORCH_COMPILE_DEFINITIONS)
    set(EXECUTORCH_CXX_STANDARD)
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
# Sets <output> to the full path, or to an empty string when the wheel does not
# carry that library.
#
# A wheel ships plain, unversioned names. That is deliberate: the library and
# the only things that link it ship in the same archive and are replaced
# together, so there is no upgrade during which two majors must coexist, and a
# versioned name would actively hurt because find_library(executorch) matches
# libexecutorch.so and not libexecutorch.so.1. The torch wheel ships unversioned
# names for the same reason.
#
# The versioned pattern is matched anyway, at no cost, so that a package
# assembled by other means than the wheel build, where the SONAME policy does
# apply and file names end in a major, still resolves. Highest major wins there
# rather than whichever sorts first.
function(_executorch_find_library _output _base_name)
  set(${_output}
      ""
      PARENT_SCOPE
  )
  # Mach-O puts the version before the suffix, libfoo.1.dylib, where ELF puts it
  # after, libfoo.so.1, so the versioned pattern differs and not just the
  # suffix.
  if(APPLE)
    file(GLOB _matches "${_executorch_package_root}/lib/${_base_name}.dylib"
         "${_executorch_package_root}/lib/${_base_name}.*.dylib"
    )
  else()
    file(GLOB _matches "${_executorch_package_root}/lib/${_base_name}.so"
         "${_executorch_package_root}/lib/${_base_name}.so.*"
    )
  endif()
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

# Computed after the lookup above rather than beside the Python extension,
# because a consumer on the older variables route is told to put this in its
# INSTALL_RPATH, and reading it before the runtime library is found handed that
# consumer an empty string.
if(_executorch_runtime_library)
  get_filename_component(
    EXECUTORCH_RUNTIME_LIBRARY_DIR "${_executorch_runtime_library}" DIRECTORY
  )
endif()

if(_executorch_runtime_library AND NOT _executorch_targets_supported)
  # The imported targets are skipped, but the libraries themselves are present
  # and linkable by path, so the long-standing variables are still honoured.
  # Leaving them empty here contradicted the message below and made a REQUIRED
  # find_package fail on a package that carries a working runtime.
  #
  # The kernel libraries are listed too, not only the runtime. The runtime
  # carries only primitive operators, not the kernels a model computes with, so
  # a consumer given the runtime alone links successfully and then fails at run
  # time with "Missing operator: aten::mul.out", which reads as a model problem
  # rather than a missing library.
  #
  # Each one is wrapped in scoped retention, because a registration-only library
  # exports nothing the application references and the linker discards it under
  # --as-needed. Measured: without this the library reaches the link line and is
  # absent from the binary's dependencies, so the registrations never run. The
  # imported targets express the same thing through link options, which this
  # older-CMake route cannot use.
  set(EXECUTORCH_FOUND ON)
  list(APPEND EXECUTORCH_LIBRARIES "${_executorch_runtime_library}")
  # Every shipped library, not only the kernels. A delegate registers itself
  # from a static initializer, so leaving one out gave a clean configure and
  # then a load failure saying the backend is not registered, which reads as a
  # model problem. Anything the wheel did not ship is simply not found and
  # skipped.
  #
  # The quantized kernels are deliberately absent, for the reason given at their
  # component definition below: they collide with the export-time plugin that
  # executorch.kernels.quantized loads, and a process holding both dies. This
  # route has no per-component target to opt into, so they are offered through
  # EXECUTORCH_QUANTIZED_KERNELS_LIBRARY instead and a consumer that wants them
  # links that as well.
  foreach(
    _executorch_component IN
    ITEMS libexecutorch_kernels_optimized
          libexecutorch_backend_xnnpack
          libexecutorch_backend_cuda
          libexecutorch_extension_cuda
          libexecutorch_backend_openvino
          libexecutorch_threadpool
          libexecutorch_etdump
  )
    _executorch_find_library(
      _executorch_component_library "${_executorch_component}"
    )
    if(_executorch_component_library)
      # The retention option only on the platform whose linker has it. Elsewhere
      # the plain path is still correct, it just leaves a registration-only
      # library subject to being dropped, which is the same position a source
      # build is in there.
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        list(
          APPEND
          EXECUTORCH_LIBRARIES
          "-Wl,--push-state,--no-as-needed,${_executorch_component_library},--pop-state"
        )
      else()
        list(APPEND EXECUTORCH_LIBRARIES "${_executorch_component_library}")
      endif()
      # The switch a source build sets. The runtime header uses it to pick
      # between an extern declaration (which resolves in the thread pool
      # library) and a local inline serial fallback. On the imported-target
      # route below it lives on the runtime target's
      # INTERFACE_COMPILE_DEFINITIONS; here it goes on
      # EXECUTORCH_COMPILE_DEFINITIONS, which is what the documented pre-3.28
      # recipe tells a consumer to apply.
      if(_executorch_component STREQUAL "libexecutorch_threadpool")
        list(APPEND EXECUTORCH_COMPILE_DEFINITIONS ET_USE_THREADPOOL)
      endif()
    endif()
  endforeach()
  unset(_executorch_component_library)
  # Held out of the aggregate above, so name it separately. A consumer that
  # wants quantized operators and does not load the Python plugin in the same
  # process links this too. Empty when the wheel shipped no such library.
  _executorch_find_library(
    EXECUTORCH_QUANTIZED_KERNELS_LIBRARY libexecutorch_kernels_quantized
  )
  if(EXECUTORCH_QUANTIZED_KERNELS_LIBRARY AND CMAKE_SYSTEM_NAME STREQUAL
                                              "Linux"
  )
    # The same scoped retention the aggregate entries get, for the same reason:
    # a registration-only library exports nothing the application references.
    set(EXECUTORCH_QUANTIZED_KERNELS_LIBRARY
        "-Wl,--push-state,--no-as-needed,${EXECUTORCH_QUANTIZED_KERNELS_LIBRARY},--pop-state"
    )
  endif()
  message(
    STATUS
      "executorch: the prebuilt runtime is present but its imported targets need CMake 3.28 or "
      "newer, because older versions write the \$ORIGIN token in a runtime search path "
      "incorrectly. EXECUTORCH_LIBRARIES carries the runtime and every shipped component by path "
      "instead. Linking it is not sufficient on its own: an imported target would also carry the "
      "include directories, the compile definitions and the C++ standard, so here a consumer has "
      "to apply EXECUTORCH_INCLUDE_DIRS, EXECUTORCH_COMPILE_DEFINITIONS and "
      "EXECUTORCH_CXX_STANDARD itself. The prebuilt Python extension is still "
      "defined, with the absolute package path only, so it links in place but is "
      "not relocatable."
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
      PROPERTIES
        IMPORTED_LOCATION "${_executorch_runtime_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
        INTERFACE_COMPILE_FEATURES cxx_std_17
        INTERFACE_COMPILE_DEFINITIONS
        "C10_USING_CUSTOM_GENERATED_MACROS;@EXECUTORCH_TRACER_DEFINITION@"
    )
    # $ORIGIN comes first so an application deployed beside its own copy of the
    # runtime finds that copy. The loader takes the first match, so leading with
    # the install directory would send a relocated application back to the
    # wheel. $ORIGIN is a loader token, so it belongs in RUNPATH and never in
    # IMPORTED_LOCATION. The package's own library directory is listed
    # explicitly because CMake records it as a link-path rpath, which
    # install(TARGETS) strips.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      # --enable-new-dtags asks for DT_RUNPATH. Without it this linker writes
      # the older DT_RPATH, which the loader searches BEFORE LD_LIBRARY_PATH and
      # applies to a dependency's dependencies too, so a consumer could not
      # point an instrumented or locally built runtime at their application.
      # Measured on GNU ld 2.35: -rpath alone produces DT_RPATH, and with this
      # flag the same link produces DT_RUNPATH.
      set_property(
        TARGET executorch::runtime
        APPEND
        PROPERTY INTERFACE_LINK_OPTIONS "LINKER:--enable-new-dtags"
                 "LINKER:-rpath,$ORIGIN" "LINKER:-rpath,$ORIGIN/../lib"
                 "LINKER:-rpath,${_executorch_package_root}/lib"
      )
    elseif(APPLE)
      # Same purpose, in Mach-O spelling. The token differs, and there is no
      # weaker older variant of the load command, so the tag selection flag has
      # no counterpart here.
      set_property(
        TARGET executorch::runtime
        APPEND
        PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,@loader_path"
                 "LINKER:-rpath,@loader_path/../lib"
                 "LINKER:-rpath,${_executorch_package_root}/lib"
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
# Call as: _executorch_define_component(<target suffix> <library base name>
# [OPT_IN])
#
# OPT_IN defines the target but keeps it out of EXECUTORCH_LIBRARIES, for a
# library a consumer has to choose deliberately rather than receive by default.
function(_executorch_define_component _suffix _library_name)
  cmake_parse_arguments(PARSE_ARGV 2 _component "OPT_IN" "" "")
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
    if(NOT _component_OPT_IN)
      set(EXECUTORCH_LIBRARIES
          ${EXECUTORCH_LIBRARIES} ${_target}
          PARENT_SCOPE
      )
    endif()
    return()
  endif()
  add_library(${_target} SHARED IMPORTED)
  set_target_properties(
    ${_target}
    PROPERTIES
      IMPORTED_LOCATION "${_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
      INTERFACE_COMPILE_FEATURES cxx_std_17
      INTERFACE_COMPILE_DEFINITIONS
      "C10_USING_CUSTOM_GENERATED_MACROS;@EXECUTORCH_TRACER_DEFINITION@"
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
  # Spelled per platform because the options and the loader relative token
  # differ. ELF takes $ORIGIN and needs a flag to choose the newer tag; Mach-O
  # takes @loader_path and has no weaker older variant to choose against. A
  # consumer configured for any other system is either cross-compiling from the
  # wrong package or has nothing to retain.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(
      TARGET ${_target}
      APPEND
      PROPERTY INTERFACE_LINK_OPTIONS
               # DT_RUNPATH rather than the older DT_RPATH, for the reason given
               # on the runtime target: DT_RPATH outranks LD_LIBRARY_PATH and
               # applies transitively, so it would stop a consumer overriding a
               # packaged library.
               "LINKER:--enable-new-dtags"
               "LINKER:-rpath,$ORIGIN"
               "LINKER:-rpath,$ORIGIN/../lib"
               # Where an in-place build finds the library, since neither token
               # above resolves to the package from a consumer's own build tree.
               # It usually arrives anyway, through the runtime this target
               # links, but that link is conditional on the runtime being
               # present while this target is not, so a package carrying a
               # component and no runtime left the component unreachable.
               # Matches the Apple branch below, which has always spelled it
               # out.
               "LINKER:-rpath,${_executorch_package_root}/lib"
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
    # Mach-O spelling of the same two search paths. There is no --no-as-needed
    # equivalent to bracket: the linker records a dependency on a dylib it was
    # given, so nothing needs to be forced to stay.
    set_property(
      TARGET ${_target}
      APPEND
      PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,@loader_path"
               "LINKER:-rpath,@loader_path/../lib"
               "LINKER:-rpath,${_executorch_package_root}/lib"
    )
  endif()
  if(NOT _component_OPT_IN)
    set(EXECUTORCH_LIBRARIES
        ${EXECUTORCH_LIBRARIES} ${_target}
        PARENT_SCOPE
    )
  endif()
endfunction()

_executorch_define_component(threadpool executorch_threadpool)

# The merged CPU kernels. Documented as a component and asserted by the release
# checks, so it has to be defined here or a consumer following the documentation
# gets a bare name that CMake hands to the linker as a literal flag.
_executorch_define_component(kernels_optimized executorch_kernels_optimized)
# The quantized kernels, optional in the same way: a wheel built without them
# simply has no such library and the component is not defined.
#
# Opt in rather than part of the aggregate. The export-time plugin that
# executorch.kernels.quantized loads registers the same operator names, and the
# runtime stops on a repeat registration rather than choosing one, so a process
# holding both dies. Measured: linking this library and importing that module in
# either order aborts with "Re-registering quantized_decomposed::add.out". None
# of the other shipped components collide this way, so only this one is held
# back, and a consumer that wants it names it.
_executorch_define_component(
  kernels_quantized executorch_kernels_quantized OPT_IN
)
# The same library exposed through a variable, so a consumer that follows the
# pre-3.28 recipe and later upgrades past 3.28 keeps working. Left empty when
# the wheel shipped no such library, matching the pre-3.28 branch above.
if(TARGET executorch::kernels_quantized)
  set(EXECUTORCH_QUANTIZED_KERNELS_LIBRARY executorch::kernels_quantized)
endif()
# The profiler. A C++ application could not record timing data from an installed
# package before, because the implementation shipped only inside the Python
# extension.
_executorch_define_component(etdump executorch_etdump)

# The switch a source build sets, on the runtime rather than on the thread pool
# target. The guarded declaration lives in a runtime header that every component
# exposes, and it selects between an extern declaration and a local inline
# definition. Putting it on the thread pool alone means a consumer with one
# translation unit linking that component and another linking only the kernels
# compiles two different definitions of the same function into one program, and
# the serial one silently wins wherever it was inlined.
#
# Only when the thread pool actually shipped. Packaging gates that library on
# the build flags it needs, so a wheel built without them has no thread pool,
# and switching the declaration on there would leave a consumer calling a
# function nothing defines.
if(TARGET executorch::runtime AND TARGET executorch::threadpool)
  set_property(
    TARGET executorch::runtime
    APPEND
    PROPERTY INTERFACE_COMPILE_DEFINITIONS ET_USE_THREADPOOL
  )
  # Published in the variable as well, so the variables route carries it too. A
  # consumer applying the variables rather than linking the target would
  # otherwise compile the serial inline fallback against a threaded library.
  if(NOT "ET_USE_THREADPOOL" IN_LIST EXECUTORCH_COMPILE_DEFINITIONS)
    list(APPEND EXECUTORCH_COMPILE_DEFINITIONS ET_USE_THREADPOOL)
  endif()
  # The definition selects a declaration, and the thread pool library holds the
  # only definition of what it declares, so a consumer linking just the runtime
  # would fail to link. Carried on the runtime rather than left to the caller,
  # since the caller cannot see which header a compile definition on an imported
  # target switched.
  set_property(
    TARGET executorch::runtime
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES executorch::threadpool
  )
endif()

_executorch_define_component(backend_xnnpack executorch_backend_xnnpack)
_executorch_define_component(backend_openvino executorch_backend_openvino)
# The CUDA delegate and its stream helper, present only in a wheel built from a
# CUDA index. A CPU wheel defines neither, so a consumer asking for one is told
# while configuring.
_executorch_define_component(backend_cuda executorch_backend_cuda)
_executorch_define_component(extension_cuda executorch_extension_cuda)

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
elseif(_executorch_runtime_library) # Tested on the located library rather than
                                    # the imported target, because the
  # targets are not defined below the CMake version they need, and a C++
  # application on an older CMake is exactly the case this branch exists to keep
  # working. A C++ application linking only the shared runtime does not need
  # Python at all, so a missing interpreter must not fail its configure. Skip
  # locating the Python extension instead; the legacy _portable_lib target is
  # simply not offered in that case.
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

if(NOT _portable_lib_LIBRARY)
  # The interpreter that answered above is whichever python3 is on PATH, which
  # is not necessarily the one this wheel was built for. A cp310 wheel inspected
  # by a 3.12 interpreter yields a suffix that names no file here, and the
  # package then reported itself as not found on a complete install. The shipped
  # extension carries its own suffix in its name, so take it from the package.
  file(GLOB _portable_lib_matches
       "${_executorch_package_root}/extension/pybindings/_portable_lib.*"
  )
  foreach(_candidate IN LISTS _portable_lib_matches)
    if(_candidate MATCHES "\\.(so|pyd|dylib)$")
      set(_portable_lib_LIBRARY "${_candidate}")
      break()
    endif()
  endforeach()
  unset(_portable_lib_matches)
endif()

if(_portable_lib_LIBRARY)
  message(
    STATUS "ExecuTorch Python extension is found at ${_portable_lib_LIBRARY}"
  )
  # Defined so that a caller who specifically wants the extension, such as a
  # custom operator project, can name the target, and deliberately kept out of
  # EXECUTORCH_LIBRARIES and out of the found decision. The extension carries
  # unresolved interpreter symbols, so a plain C++ application that links it
  # fails with a page of PyUnicode_InternFromString style errors. Offering it as
  # find_package's answer meant a wheel with no linkable library configured
  # cleanly and failed at link instead of saying so, and it no longer stands in
  # for a fused layout: every platform that ships a runtime now ships it as its
  # own file.
  #
  # The target carries the C++20 requirement PyTorch's headers need, while the
  # runtime components require C++17.
  if(TARGET _portable_lib)
    # This file ran already in the same configure, because another subproject
    # called find_package too. No in-tree target uses this name, so it can only
    # be the imported one defined below, and re-setting its properties to the
    # same values is harmless.
    message(STATUS "executorch: _portable_lib is already defined, reusing it")
  else()
    add_library(_portable_lib STATIC IMPORTED)
  endif()
  # PyTorch requires C++20, so pybindings must be compiled with C++20.
  set_target_properties(
    _portable_lib
    PROPERTIES
      IMPORTED_LOCATION "${_portable_lib_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
      # An interface requirement rather than CXX_STANDARD: an imported
      # target compiles nothing itself, and CXX_STANDARD does not reach
      # consumers, so a custom-op build linking this could still
      # compile
      # as C++17 and fail against headers that need C++20.
      INTERFACE_COMPILE_FEATURES cxx_std_20
      # The runtime's definitions. A custom-op build that links only this target
      # compiles against the same headers and needs them too. The thread pool
      # definition is appended after this call rather than listed here, because
      # whether it applies depends on a variable set further up.
      INTERFACE_COMPILE_DEFINITIONS
      "C10_USING_CUSTOM_GENERATED_MACROS;@EXECUTORCH_TRACER_DEFINITION@"
  )
  # Appended here rather than listed above, because whether the thread pool
  # definition applies depends on whether this wheel shipped it. Without it a
  # consumer compiles the serial inline copies of parallel_for while the shipped
  # libraries carry the real ones, and the serial version wins wherever it was
  # inlined, with no diagnostic. It does not arrive transitively because the
  # runtime is attached here as a file path, not as the target.
  set(_executorch_extension_needs_threadpool OFF)
  if("ET_USE_THREADPOOL" IN_LIST EXECUTORCH_COMPILE_DEFINITIONS)
    set(_executorch_extension_needs_threadpool ON)
  endif()
  if(_executorch_extension_needs_threadpool)
    set_property(
      TARGET _portable_lib
      APPEND
      PROPERTY INTERFACE_COMPILE_DEFINITIONS ET_USE_THREADPOOL
    )
  endif()
  # The extension links the runtime rather than containing it, so it no longer
  # satisfies the runtime symbols a custom-op library references. Put the
  # shipped runtime on this target's interface, which is where the definitions
  # moved to, so an out-of-tree operator project keeps building and loading
  # against the extension exactly as it did before. Without this a custom
  # operator links and then fails to load with an undefined runtime symbol.
  #
  # The file path rather than executorch::runtime, because that target is only
  # defined on CMake 3.28 or newer while this one is defined deliberately on
  # every version, so a consumer on an older CMake can still link the extension.
  if(_executorch_runtime_library)
    set_property(
      TARGET _portable_lib
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES "${_executorch_runtime_library}"
    )
    # CMake adds a linked library's directory to the consumer's build tree
    # runtime search path and strips it on install, so an installed consumer
    # library reports libexecutorch.so as not found. Publish the directories
    # rather than forcing them onto the target: an interface link option reaches
    # every consumer and survives install, which would bake this machine's
    # package location into a library the consumer ships onward. A consumer that
    # installs elsewhere adds these to its own INSTALL_RPATH.
  endif()
endif()

# find_package checks <package name>_FOUND, which is case-sensitive and does not
# match the EXECUTORCH_FOUND spelling this file documents. Without this, a
# REQUIRED find_package would succeed even when nothing usable was located.
set(executorch_FOUND ${EXECUTORCH_FOUND})
if(NOT executorch_FOUND)
  # The reason, not an error: the single REQUIRED gate at the bottom of this
  # file raises. A component request that fails replaces this with a message
  # naming the component, which is the more specific answer to what was asked
  # for.
  #
  # One string rather than several arguments. Several make a list, and message()
  # joins a list with semicolons, which lands separators mid sentence.
  string(
    CONCAT
      executorch_NOT_FOUND_MESSAGE
      "this ExecuTorch package ships the headers but no linkable library. The wheel "
      "for this platform carries the Python extension only, which cannot stand in for "
      "the runtime because it references the interpreter, so a C++ consumer has "
      "nothing to link against"
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
      set(EXECUTORCH_FOUND OFF)
      set(executorch_FOUND FALSE)
      # Naming the CMake version when that is the cause saves a consumer from
      # concluding the component is missing from the package, which is the wrong
      # thing to go looking for.
      if(NOT _executorch_targets_supported)
        # One string rather than several arguments. Several make a list, and
        # message() joins a list with semicolons, which lands separators mid
        # sentence.
        #
        # The quantized kernels are held out of EXECUTORCH_LIBRARIES on purpose,
        # so a consumer who wants them names
        # EXECUTORCH_QUANTIZED_KERNELS_LIBRARY instead. See the OPT_IN comment
        # at the component definition above.
        if(_component STREQUAL "kernels_quantized")
          string(
            CONCAT
              executorch_NOT_FOUND_MESSAGE
              "the required component '${_component}' needs CMake 3.28 or newer, because "
              "older versions write the \$ORIGIN token in a runtime search path incorrectly; "
              "this package is otherwise usable through EXECUTORCH_QUANTIZED_KERNELS_LIBRARY"
          )
        else()
          string(
            CONCAT
              executorch_NOT_FOUND_MESSAGE
              "the required component '${_component}' needs CMake 3.28 or newer, because older "
              "versions write the \$ORIGIN token in a runtime search path incorrectly; this "
              "package is otherwise usable through EXECUTORCH_LIBRARIES"
          )
        endif()
      else()
        # One string rather than several arguments. Several make a list, and
        # message() joins a list with semicolons, which lands separators mid
        # sentence.
        string(
          CONCAT
            executorch_NOT_FOUND_MESSAGE
            "this ExecuTorch package does not provide the required component "
            "'${_component}'. The installed wheel does not contain the library that "
            "component wraps, either because it was not built with it or because this "
            "platform's wheel ships the headers without the separate libraries"
        )
      endif()
    endif()
  endif()
endforeach()
if(NOT executorch_FOUND AND executorch_FIND_REQUIRED)
  message(FATAL_ERROR "${executorch_NOT_FOUND_MESSAGE}")
endif()
