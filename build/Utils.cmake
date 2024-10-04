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
      "  EXECUTORCH_ENABLE_LOGGING              : ${EXECUTORCH_ENABLE_LOGGING}"
  )
  message(STATUS "  EXECUTORCH_ENABLE_PROGRAM_VERIFICATION : "
                 "${EXECUTORCH_ENABLE_PROGRAM_VERIFICATION}"
  )
  message(
    STATUS "  EXECUTORCH_LOG_LEVEL                   : ${EXECUTORCH_LOG_LEVEL}"
  )
  message(STATUS "  EXECUTORCH_BUILD_ANDROID_JNI           : "
                 "${EXECUTORCH_BUILD_ANDROID_JNI}"
  )
  message(STATUS "  EXECUTORCH_BUILD_ARM_BAREMETAL         : "
                 "${EXECUTORCH_BUILD_ARM_BAREMETAL}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_COREML                : ${EXECUTORCH_BUILD_COREML}"
  )
  message(STATUS "  EXECUTORCH_BUILD_KERNELS_CUSTOM        : "
                 "${EXECUTORCH_BUILD_KERNELS_CUSTOM}"
  )
  message(STATUS "  EXECUTORCH_BUILD_EXECUTOR_RUNNER       : "
                 "${EXECUTORCH_BUILD_EXECUTOR_RUNNER}"
  )
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_DATA_LOADER : "
                 "${EXECUTORCH_BUILD_EXTENSION_DATA_LOADER}"
  )
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_MODULE      : "
                 "${EXECUTORCH_BUILD_EXTENSION_MODULE}"
  )
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL : "
                 "${EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL}"
  )
  message(STATUS "  EXECUTORCH_BUILD_EXTENSION_TENSOR      : "
                 "${EXECUTORCH_BUILD_EXTENSION_TENSOR}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_FLATC                 : ${EXECUTORCH_BUILD_FLATC}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_GFLAGS                : ${EXECUTORCH_BUILD_GFLAGS}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_GTESTS                : ${EXECUTORCH_BUILD_GTESTS}"
  )
  message(STATUS "  EXECUTORCH_BUILD_HOST_TARGETS          : "
                 "${EXECUTORCH_BUILD_HOST_TARGETS}"
  )
  message(
    STATUS "  EXECUTORCH_BUILD_MPS                   : ${EXECUTORCH_BUILD_MPS}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_PYBIND                : ${EXECUTORCH_BUILD_PYBIND}"
  )
  message(
    STATUS "  EXECUTORCH_BUILD_QNN                   : ${EXECUTORCH_BUILD_QNN}"
  )
  message(STATUS "  EXECUTORCH_BUILD_KERNELS_OPTIMIZED     : "
                 "${EXECUTORCH_BUILD_KERNELS_OPTIMIZED}"
  )
  message(STATUS "  EXECUTORCH_BUILD_KERNELS_QUANTIZED     : "
                 "${EXECUTORCH_BUILD_KERNELS_QUANTIZED}"
  )
  message(
    STATUS "  EXECUTORCH_BUILD_DEVTOOLS              : ${EXECUTORCH_BUILD_DEVTOOLS}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_SIZE_TEST             : ${EXECUTORCH_BUILD_SIZE_TEST}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_XNNPACK               : ${EXECUTORCH_BUILD_XNNPACK}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_VULKAN                : ${EXECUTORCH_BUILD_VULKAN}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_PTHREADPOOL           : ${EXECUTORCH_BUILD_PTHREADPOOL}"
  )
  message(
    STATUS
      "  EXECUTORCH_BUILD_CPUINFO               : ${EXECUTORCH_BUILD_CPUINFO}"
  )

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
    LINKER:--no-whole-archive"
  )
endfunction()

# Same as kernel_link_options but it's for MacOS linker
function(macos_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:-force_load,$<TARGET_FILE:${target_name}>"
  )
endfunction()

# Same as kernel_link_options but it's for MSVC linker
function(msvc_kernel_link_options target_name)
  target_link_options(
    ${target_name} INTERFACE
    "SHELL:LINKER:/WHOLEARCHIVE:$<TARGET_FILE:${target_name}>"
  )
endfunction()

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
function(target_link_options_shared_lib target_name)
  if(APPLE)
    macos_kernel_link_options(${target_name})
  elseif(MSVC)
    msvc_kernel_link_options(${target_name})
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
        --config=${executorch_root}/build/cmake_deps.toml --out=${sources_file}
        --buck2=${BUCK2}
      OUTPUT_VARIABLE gen_srcs_output
      ERROR_VARIABLE gen_srcs_error
      RESULT_VARIABLE gen_srcs_exit_code
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(NOT gen_srcs_exit_code EQUAL 0)
      message("Error while generating ${sources_file}. "
              "Exit code: ${gen_srcs_exit_code}"
      )
      message("Output:\n${gen_srcs_output}")
      message("Error:\n${gen_srcs_error}")
      message(FATAL_ERROR "executorch: source list generation failed")
    endif()
  endif()
endfunction()

# Sets the value of the BUCK2 variable by searching for a buck2 binary with the
# correct version.
#
# The resolve_buck.py script uses the following logic to find buck2: 1) If BUCK2
# argument is set explicitly, use it. Warn if the version is incorrect. 2) Look
# for a binary named buck2 on the system path. Take it if it is the correct
# version. 3) Check for a previously downloaded buck2 binary (from step 4). 4)
# Download and cache correct version of buck2.
function(resolve_buck2)
  if(EXECUTORCH_ROOT)
    set(executorch_root ${EXECUTORCH_ROOT})
  else()
    set(executorch_root ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  set(resolve_buck2_command
      ${PYTHON_EXECUTABLE} ${executorch_root}/build/resolve_buck.py
      --cache_dir=${CMAKE_CURRENT_BINARY_DIR}/buck2-bin
  )

  if(NOT ${BUCK2} STREQUAL "")
    list(APPEND resolve_buck2_command --buck2=${BUCK2})
  endif()

  execute_process(
    COMMAND ${resolve_buck2_command}
    OUTPUT_VARIABLE resolve_buck2_output
    ERROR_VARIABLE resolve_buck2_error
    RESULT_VARIABLE resolve_buck2_exit_code
    WORKING_DIRECTORY ${executorch_root}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # $BUCK2 is a copy of the var from the parent scope. This block will set
  # $buck2 to the value we want to return.
  if(resolve_buck2_exit_code EQUAL 0)
    set(buck2 ${resolve_buck2_output})
    message(STATUS "Resolved buck2 as ${resolve_buck2_output}.")
  elseif(resolve_buck2_exit_code EQUAL 2)
    # Wrong buck version used. Stop here to ensure that the user sees the error.
    message(FATAL_ERROR "Failed to resolve buck2.\n${resolve_buck2_error}")
  else()
    # Unexpected failure of the script. Warn.
    message(WARNING "Failed to resolve buck2.")
    message(WARNING "${resolve_buck2_error}")

    if("${BUCK2}" STREQUAL "")
      set(buck2 "buck2")
    endif()
  endif()

  # Update the var in the parent scope. Note that this does not modify our
  # local $BUCK2 value.
  set(BUCK2 "${buck2}" PARENT_SCOPE)

  # The buck2 daemon can get stuck. Killing it can help.
  message(STATUS "Killing buck2 daemon")
  execute_process(
    # Note that we need to use the local buck2 variable. BUCK2 is only set in
    # the parent scope, and can still be empty in this scope.
    COMMAND "${buck2} kill"
    WORKING_DIRECTORY ${executorch_root}
    COMMAND_ECHO STDOUT
  )
endfunction()

# Sets the value of the PYTHON_EXECUTABLE variable to 'python' if in an active
# (non-base) conda environment, and 'python3' otherwise. This maintains
# backwards compatibility for non-conda users and avoids conda users needing to
# explicitly set PYTHON_EXECUTABLE=python.
function(resolve_python_executable)
  # Counter-intuitively, CONDA_DEFAULT_ENV contains the name of the active
  # environment.
  if(DEFINED ENV{CONDA_DEFAULT_ENV} AND NOT $ENV{CONDA_DEFAULT_ENV} STREQUAL
                                        "base"
  )
    set(PYTHON_EXECUTABLE
        python
        PARENT_SCOPE
    )
  else()
    set(PYTHON_EXECUTABLE
        python3
        PARENT_SCOPE
    )
  endif()
endfunction()
