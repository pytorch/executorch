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

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(executorch_target_link_options_shared_lib target_name)
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
