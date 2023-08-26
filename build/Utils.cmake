# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is intended to have helper functions to keep the CMakeLists.txt
# concise. If there are any helper function can be re-used, it's recommented to
# add them here.

# Public function to print summary for all configurations. For new variable,
# it's recommended to add them here.
function(executorch_print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "  BUCK                          : ${BUCK2}")
  message(STATUS "  CMAKE_CXX_STANDARD            : ${CMAKE_CXX_STANDARD}")
  message(STATUS "  CMAKE_CXX_COMPILER_ID         : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "  CMAKE_TOOLCHAIN_FILE          : ${CMAKE_TOOLCHAIN_FILE}")
  message(STATUS "  FLATBUFFERS_BUILD_FLATC       : ${FLATBUFFERS_BUILD_FLATC}")
  message(
    STATUS "  FLATBUFFERS_BUILD_FLATHASH    : ${FLATBUFFERS_BUILD_FLATHASH}")
  message(
    STATUS "  FLATBUFFERS_BUILD_FLATLIB     : ${FLATBUFFERS_BUILD_FLATLIB}")
  message(STATUS "  FLATBUFFERS_BUILD_TESTS       : ${FLATBUFFERS_BUILD_TESTS}")
  message(
    STATUS "  REGISTER_EXAMPLE_CUSTOM_OPS   : ${REGISTER_EXAMPLE_CUSTOM_OPS}")
endfunction()

# This is the funtion to use -Wl, --whole-archive to link static library NB:
# target_link_options is broken for this case, it only append the interface link
# options of the first library.
function(kernel_link_options target_name)
  # target_link_options(${target_name} INTERFACE
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,target_name>")
  target_link_options(
    ${target_name}
    INTERFACE
    "SHELL:LINKER:--whole-archive $<TARGET_FILE:${target_name}> LINKER:--no-whole-archive"
  )
endfunction()

function(macos_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    # Same as kernel_link_options but it's for MacOS linker
    "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>")
endfunction()

function(target_link_options_shared_lib target_name)
  # Ensure that the load-time constructor functions run. By default, the linker
  # would remove them since there are no other references to them.
  if(APPLE)
    macos_kernel_link_options(${target_name})
  else()
    kernel_link_options(${target_name})
  endif()
endfunction()

# Extract source files based on toml config. This is useful to keep buck2 and
# cmake aligned.
function(extract_sources sources_file)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} build/extract_sources.py --buck2=${BUCK2}
            --config=build/cmake_deps.toml --out=${sources_file}
    OUTPUT_VARIABLE gen_srcs_output
    ERROR_VARIABLE gen_srcs_error
    RESULT_VARIABLE gen_srcs_exit_code
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT gen_srcs_exit_code EQUAL 0)
    message("Error while generating ${sources_file}. "
            "Exit code: ${gen_srcs_exit_code}")
    message("Output:\n${gen_srcs_output}")
    message("Error:\n${gen_srcs_error}")
    message(FATAL_ERROR "executorch: source list generation failed")
  endif()
endfunction()
