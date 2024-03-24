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
# cmake-format --first-comment-is-literal=True Utils.cmake
# ~~~
# It should also be cmake-lint clean.
#

# Public function to print summary for all configurations. For new variables,
# it's recommended to add them here.
function(executorch_print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "  CMAKE_BUILD_TYPE              : ${CMAKE_BUILD_TYPE}")
  message(STATUS "  CMAKE_CXX_STANDARD            : ${CMAKE_CXX_STANDARD}")
  message(STATUS "  CMAKE_CXX_COMPILER_ID         : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "  CMAKE_TOOLCHAIN_FILE          : ${CMAKE_TOOLCHAIN_FILE}")
  message(STATUS "  BUCK2                         : ${BUCK2}")
  message(STATUS "  PYTHON_EXECUTABLE             : ${PYTHON_EXECUTABLE}")
  message(STATUS "  FLATC_EXECUTABLE              : ${FLATC_EXECUTABLE}")
  message(
    STATUS
      "  EXECUTORCH_ENABLE_LOGGING              : ${EXECUTORCH_ENABLE_LOGGING}")
  message(STATUS "  EXECUTORCH_ENABLE_PROGRAM_VERIFICATION : "
                 "${EXECUTORCH_ENABLE_PROGRAM_VERIFICATION}")
  message(
    STATUS "  EXECUTORCH_LOG_LEVEL                   : ${EXECUTORCH_LOG_LEVEL}")
  message(STATUS "  EXECUTORCH_BUILD_ANDROID_JNI           : "
                 "${EXECUTORCH_BUILD_ANDROID_JNI}")
  message(STATUS "  EXECUTORCH_BUILD_ARM_BAREMETAL         : "
                 "${EXECUTORCH_BUILD_ARM_BAREMETAL}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_COREML                : ${EXECUTORCH_BUILD_COREML}")
  message(STATUS "  EXECUTORCH_BUILD_EXECUTOR_RUNNER       : "
                 "${EXECUTORCH_BUILD_EXECUTOR_RUNNER}")
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_AOT_UTIL    : "
                 "${EXECUTORCH_BUILD_EXTENSION_AOT_UTIL}")
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER : "
                 "${EXECUTORCH_BUILD_EXTENSION_DATA_LOADER}")
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_MODULE      : "
                 "${EXECUTORCH_BUILD_EXTENSION_MODULE}")
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL : "
                 "${EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_FLATC                 : ${EXECUTORCH_BUILD_FLATC}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_GFLAGS                : ${EXECUTORCH_BUILD_GFLAGS}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_GTESTS                : ${EXECUTORCH_BUILD_GTESTS}")
  message(STATUS "  EXECUTORCH_BUILD_HOST_TARGETS          : "
                 "${EXECUTORCH_BUILD_HOST_TARGETS}")
  message(
    STATUS "  EXECUTORCH_BUILD_MPS                   : ${EXECUTORCH_BUILD_MPS}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_PYBIND                : ${EXECUTORCH_BUILD_PYBIND}")
  message(
    STATUS "  EXECUTORCH_BUILD_QNN                   : ${EXECUTORCH_BUILD_QNN}")
  message(
    STATUS "  EXECUTORCH_REGISTER_OPTIMIZED_OPS      : ${EXECUTORCH_REGISTER_OPTIMIZED_OPS}")
  message(
    STATUS "  EXECUTORCH_REGISTER_QUANTIZED_OPS      : ${EXECUTORCH_REGISTER_QUANTIZED_OPS}")
  message(
    STATUS "  EXECUTORCH_BUILD_SDK                   : ${EXECUTORCH_BUILD_SDK}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_SIZE_TEST             : ${EXECUTORCH_BUILD_SIZE_TEST}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_XNNPACK               : ${EXECUTORCH_BUILD_XNNPACK}")
  message(
    STATUS
      "  EXECUTORCH_BUILD_VULKAN                : ${EXECUTORCH_BUILD_VULKAN}")
endfunction()

# This is the funtion to use -Wl, --whole-archive to link static library NB:
# target_link_options is broken for this case, it only append the interface link
# options of the first library.
function(kernel_link_options target_name)
  # target_link_options(${target_name} INTERFACE
  # "$<LINK_LIBRARY:WHOLE_ARCHIVE,target_name>")
  target_link_options(
    ${target_name} INTERFACE "SHELL:LINKER:--whole-archive \
    $<TARGET_FILE:${target_name}> \
    LINKER:--no-whole-archive")
endfunction()

# Same as kernel_link_options but it's for MacOS linker
function(macos_kernel_link_options target_name)
  target_link_options(${target_name} INTERFACE
                      "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>")
endfunction()

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(target_link_options_shared_lib target_name)
  if(APPLE)
    macos_kernel_link_options(${target_name})
  else()
    kernel_link_options(${target_name})
  endif()
endfunction()

# Extract source files based on toml config. This is useful to keep buck2 and
# cmake aligned. Do not regenerate if file exists.
function(extract_sources sources_file)
  if(EXISTS "${sources_file}")
    message(STATUS "executorch: Using source file list ${sources_file}")
  else()
    # A file wasn't generated. Run a script to extract the source lists from the
    # buck2 build system and write them to a file we can include.
    #
    # NOTE: This will only happen once during cmake setup, so it will not re-run
    # if the buck2 targets change.
    message(STATUS "executorch: Generating source file list ${sources_file}")
    if(EXECUTORCH_ROOT)
      set(executorch_root ${EXECUTORCH_ROOT})
    else()
      set(executorch_root ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    execute_process(
      COMMAND
        ${PYTHON_EXECUTABLE} ${executorch_root}/build/extract_sources.py
        --buck2=${BUCK2} --config=${executorch_root}/build/cmake_deps.toml
        --out=${sources_file}
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
  endif()
endfunction()
