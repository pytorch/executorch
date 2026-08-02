# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# This file is intended to have helper functions to keep the CMakeLists.txt
# concise. If there are any helper function can be re-used, it's recommented to
# add them here.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i Utils.cmake
# ~~~
# It should also be cmake-lint clean.
#

# This is the funtion to use -Wl, --whole-archive to link static library NB:
# target_link_options is broken for this case, it only append the interface link
# options of the first library.
function(executorch_kernel_link_options target_name)
  # target_link_options(${target_name} INTERFACE
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,target_name>")
  target_link_options(
    ${target_name} INTERFACE "SHELL:LINKER:--whole-archive \
    $<TARGET_FILE:${target_name}> \
    LINKER:--no-whole-archive"
  )
endfunction()

# Same as executorch_kernel_link_options but it's for MacOS linker
function(executorch_macos_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>"
  )
endfunction()

# Same as executorch_kernel_link_options but it's for MSVC linker
function(executorch_msvc_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:/WHOLEARCHIVE:$<TARGET_FILE:${target_name}>"
  )
endfunction()

# Add a whole-archive reference to a static library on a consumer's link line.
#
# This is deliberately a link option rather than a link library: CMake refuses
# to mix the WHOLE_ARCHIVE link feature with the plain references other targets
# make to the same archive, and link options are also emitted before the ordered
# link libraries, which is what keeps a bundled archive ahead of anything that
# would otherwise satisfy the same symbols.
function(executorch_target_whole_archive target_name archive_target)
  if(APPLE)
    set(_flags "SHELL:LINKER:-force_load,$<TARGET_FILE:${archive_target}>")
  elseif(MSVC)
    set(_flags "SHELL:LINKER:/WHOLEARCHIVE:$<TARGET_FILE:${archive_target}>")
  else()
    set(_flags
        "SHELL:LINKER:--whole-archive $<TARGET_FILE:${archive_target}> LINKER:--no-whole-archive"
    )
  endif()
  target_link_options(${target_name} PRIVATE "${_flags}")
  add_dependencies(${target_name} ${archive_target})
endfunction()

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(executorch_target_link_options_shared_lib target_name)
  # A shared library cannot be retained with --whole-archive: that flag only
  # governs how an archive's members are pulled in, so the library is still
  # subject to --as-needed and gets dropped along with its registration
  # constructor. Export scoped --no-as-needed retention instead, which is what
  # actually keeps a registration-only shared library on the link line.
  get_target_property(_target_type ${target_name} TYPE)
  if(_target_type STREQUAL "SHARED_LIBRARY" AND NOT (APPLE OR MSVC))
    target_link_options(
      ${target_name}
      INTERFACE
      # Separate options rather than one SHELL: string, which splits on spaces
      # and would break a library path containing one.
      "LINKER:--push-state,--no-as-needed"
      "$<TARGET_FILE:${target_name}>"
      "LINKER:--pop-state"
    )
    return()
  endif()
  if(APPLE)
    executorch_macos_kernel_link_options(${target_name})
  elseif(MSVC)
    executorch_msvc_kernel_link_options(${target_name})
  else()
    executorch_kernel_link_options(${target_name})
  endif()
endfunction()

function(target_link_options_gc_sections target_name)
  if(APPLE)
    target_link_options(${target_name} PRIVATE "LINKER:-dead_strip")
  elseif(WIN32)
    target_link_options(${target_name} PRIVATE "LINKER:/OPT:REF")
  else()
    target_link_options(${target_name} PRIVATE "LINKER:--gc-sections")
  endif()
endfunction()

function(resolve_python_executable)
  if(NOT PYTHON_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(PYTHON_EXECUTABLE
        ${Python3_EXECUTABLE}
        PARENT_SCOPE
    )
  endif()
endfunction()

# find_package(Torch CONFIG REQUIRED) replacement for targets that have a
# header-only Torch dependency.
#
# Unlike find_package(Torch ...), this will only set TORCH_INCLUDE_DIRS in the
# parent scope. In particular, it will NOT set any of the following: -
# TORCH_FOUND - TORCH_LIBRARY - TORCH_CXX_FLAGS
function(find_package_torch_headers)
  # We implement this way rather than using find_package so that
  # cross-compilation can still use the host's installed copy of torch, since
  # the headers should be fine.
  get_torch_base_path(TORCH_BASE_PATH)
  set(TORCH_INCLUDE_DIRS
      "${TORCH_BASE_PATH}/include;${TORCH_BASE_PATH}/include/torch/csrc/api/include"
      PARENT_SCOPE
  )
endfunction()

# Return the base path to the installed Torch Python library in outVar.
function(get_torch_base_path outVar)
  if(NOT PYTHON_EXECUTABLE)
    resolve_python_executable()
  endif()
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import importlib.util; print(importlib.util.find_spec('torch').submodule_search_locations[0])"
    OUTPUT_VARIABLE _tmp_torch_path
    ERROR_VARIABLE _tmp_torch_path_error
    RESULT_VARIABLE _tmp_torch_path_result COMMAND_ECHO STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _tmp_torch_path_result EQUAL 0)
    message("Error while adding torch to CMAKE_PREFIX_PATH. "
            "Exit code: ${_tmp_torch_path_result}"
    )
    message("Output:\n${_tmp_torch_path}")
    message(FATAL_ERROR "Error:\n${_tmp_torch_path_error}")
  endif()
  set(${outVar}
      ${_tmp_torch_path}
      PARENT_SCOPE
  )
endfunction()

# Add the Torch CMake configuration to CMAKE_PREFIX_PATH so that find_package
# can find Torch.
function(add_torch_to_cmake_prefix_path)
  get_torch_base_path(_tmp_torch_path)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_torch_path}")
  set(CMAKE_PREFIX_PATH
      "${CMAKE_PREFIX_PATH}"
      PARENT_SCOPE
  )
endfunction()

# Replacement for find_package(Torch CONFIG REQUIRED); sets up CMAKE_PREFIX_PATH
# first and only does the find once. If you have a header-only Torch dependency,
# use find_package_torch_headers instead!
macro(find_package_torch)
  if(NOT TARGET torch)
    add_torch_to_cmake_prefix_path()
    find_package(Torch CONFIG REQUIRED)
  endif()
endmacro()

# Modify ${targetName}'s INTERFACE_INCLUDE_DIRECTORIES by wrapping each entry in
# $<BUILD_INTERFACE:...> so that they work with CMake EXPORT.
function(executorch_move_interface_include_directories_to_build_time_only
         targetName
)
  get_property(
    OLD_INTERFACE_INCLUDE_DIRECTORIES
    TARGET "${targetName}"
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  )
  set(FIXED_INTERFACE_INCLUDE_DIRECTORIES)
  foreach(dir ${OLD_INTERFACE_INCLUDE_DIRECTORIES})
    list(APPEND FIXED_INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${dir}>)
  endforeach()
  set_property(
    TARGET "${targetName}" PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                    ${FIXED_INTERFACE_INCLUDE_DIRECTORIES}
  )
endfunction()

function(executorch_add_prefix_to_public_headers targetName prefix)
  get_property(
    OLD_PUBLIC_HEADERS
    TARGET "${targetName}"
    PROPERTY PUBLIC_HEADER
  )
  set(FIXED_PUBLIC_HEADERS)
  foreach(header ${OLD_PUBLIC_HEADERS})
    list(APPEND FIXED_PUBLIC_HEADERS "${prefix}${header}")
  endforeach()
  set_property(
    TARGET "${targetName}" PROPERTY PUBLIC_HEADER ${FIXED_PUBLIC_HEADERS}
  )
endfunction()

# -----------------------------------------------------------------------------
# MLX metallib distribution helper
# -----------------------------------------------------------------------------
# Copies mlx.metallib next to the target executable so MLX can find it at
# runtime.
#
# MLX uses dladdr() to find the directory containing the binary with MLX code,
# then looks for mlx.metallib in that directory. When MLX is statically linked
# into an executable or shared library, this function ensures the metallib is
# colocated with that binary.
#
# Usage: executorch_target_copy_mlx_metallib(my_executable)
#
function(executorch_target_copy_mlx_metallib target)
  if(EXECUTORCH_BUILD_MLX)
    if(DEFINED MLX_METALLIB_PATH AND EXISTS "${MLX_METALLIB_PATH}")
      add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MLX_METALLIB_PATH}"
                "$<TARGET_FILE_DIR:${target}>/mlx.metallib"
        COMMENT "Copying mlx.metallib for ${target}"
      )
    elseif(DEFINED MLX_METALLIB_PATH)
      message(
        WARNING
          "MLX_METALLIB_PATH is set to ${MLX_METALLIB_PATH} but file does not exist. "
          "metallib will not be copied for ${target}."
      )
    endif()
  endif()
endfunction()

# Make a target resolve the ExecuTorch runtime from libexecutorch.so.
#
# Naming the shared runtime as an ordinary dependency is not enough. CMake
# orders link libraries so that an archive precedes what it depends on, which
# puts libexecutorch_core.a ahead of libexecutorch.so; the archive then
# satisfies the runtime symbols first and the target ends up with a private copy
# of the backend registry. Link options come before the ordered libraries, so
# naming the runtime there leaves the archive with nothing left to resolve.
#
# On ELF platforms --no-as-needed is needed around it, because a shared library
# with no already-referenced symbol at the point it appears can be dropped, and
# the static archive further along the line would then supply the registry after
# all. Other linkers keep the reference without it.
function(executorch_target_link_shared_runtime target_name)
  executorch_target_retain_shared_library(${target_name} executorch_shared)
endfunction()

# Put a shared library on a consumer's link line and keep it there.
#
# A library whose only purpose is to run a static initializer, such as a backend
# or an operator registration library, has no symbol the consumer references
# directly, so the linker is free to drop it from DT_NEEDED. Some linkers do
# exactly that and the initializer never runs, which shows up at runtime as a
# backend or kernel that is missing rather than as a link error.
function(executorch_target_retain_shared_library target_name library_target)
  if(NOT EXECUTORCH_BUILD_SHARED)
    return()
  endif()
  if(APPLE OR MSVC)
    # TARGET_LINKER_FILE rather than TARGET_FILE: on Windows the linker needs
    # the import library, not the DLL itself. Plain rather than SHELL: this is a
    # single path, and SHELL splits on spaces, so a path containing one would
    # reach the linker as two broken arguments.
    set(_retain_flags "$<TARGET_LINKER_FILE:${library_target}>")
  else()
    # push-state/pop-state rather than closing with an explicit --as-needed:
    # that would leave --as-needed in force for everything after it on the line
    # and drop the next library that only exists for static-init registration.
    # Separate options rather than one SHELL: string, which splits on spaces and
    # would break a library path containing one.
    set(_retain_flags "LINKER:--push-state,--no-as-needed"
                      "$<TARGET_FILE:${library_target}>" "LINKER:--pop-state"
    )
  endif()
  # The generator expression alone does not order the build, so say it outright.
  add_dependencies(${target_name} ${library_target})
  set_property(
    TARGET ${target_name}
    APPEND
    PROPERTY LINK_OPTIONS "${_retain_flags}"
  )
endfunction()

# Create and install a shared library composed from dependency libraries. The
# target links the provided dependencies and carries VERSION/SOVERSION.
function(executorch_add_shared_library target_name)
  set(empty_source_name "${target_name}_empty.cpp")
  file(
    GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${empty_source_name}"
    CONTENT "// intentionally empty\n"
  )
  add_library(
    ${target_name} SHARED "${CMAKE_CURRENT_BINARY_DIR}/${empty_source_name}"
  )
  if(ARGN)
    target_link_libraries(${target_name} PRIVATE ${ARGN})
  endif()
  set_target_properties(
    ${target_name}
    PROPERTIES VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               LINKER_LANGUAGE CXX
  )
  install(
    TARGETS ${target_name}
    EXPORT ExecuTorchTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endfunction()
