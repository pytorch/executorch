# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_executor_runner LANGUAGES C CXX)

get_filename_component(
  _default_executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE
)

if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT
      "${_default_executorch_root}"
      CACHE PATH "Path to an ExecuTorch checkout"
  )
endif()

if(NOT EXISTS "${EXECUTORCH_ROOT}/CMakeLists.txt")
  if(EXISTS "${_default_executorch_root}/CMakeLists.txt")
    message(
      WARNING
        "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project. Falling back to ${_default_executorch_root}."
    )
    set(EXECUTORCH_ROOT
        "${_default_executorch_root}"
        CACHE PATH "Path to an ExecuTorch checkout" FORCE
    )
  else()
    message(
      FATAL_ERROR
        "EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) does not contain an ExecuTorch CMake project."
    )
  endif()
endif()

set(ARM_EXECUTOR_RUNNER_STANDALONE
    ON
    CACHE BOOL
          "Indicates arm_executor_runner was configured as a standalone project"
          FORCE
)

# Load the preset helper so standalone builds inherit the same defaults as the
# superbuild (toolchains, delegated targets, devtools options, etc.).
set(_executorch_preset_cmake
    "${EXECUTORCH_ROOT}/tools/cmake/common/preset.cmake"
)
if(EXISTS "${_executorch_preset_cmake}")
  include("${_executorch_preset_cmake}")
  if(NOT DEFINED EXECUTORCH_BUILD_PRESET_FILE)
    set(EXECUTORCH_BUILD_PRESET_FILE
        "${EXECUTORCH_ROOT}/tools/cmake/preset/arm_baremetal.cmake"
        CACHE PATH "Preset used when configuring the standalone runner"
    )
  endif()
  load_build_preset()
endif()
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

option(ARM_EXECUTOR_RUNNER_SKIP_INSTALL_RULES
       "Skip install() rules for standalone arm_executor_runner builds" ON
)
if(DEFINED CMAKE_SKIP_INSTALL_RULES)
  set(_arm_runner_skip_install_rules "${CMAKE_SKIP_INSTALL_RULES}")
endif()
if(ARM_EXECUTOR_RUNNER_SKIP_INSTALL_RULES)
  set(CMAKE_SKIP_INSTALL_RULES ON)
endif()

foreach(
  _opt
  EXECUTORCH_BUILD_ARM_BAREMETAL EXECUTORCH_BUILD_CORTEX_M
  EXECUTORCH_BUILD_KERNELS_QUANTIZED EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL
)
  if(NOT DEFINED ${_opt})
    set(${_opt}
        ON
        CACHE BOOL "" FORCE
    )
  endif()
endforeach()
set(EXECUTORCH_SKIP_ARM_EXECUTOR_RUNNER
    ON
    CACHE BOOL "" FORCE
)

# examples/arm/executor_runner/CMakeLists.txt generates the runner-specific
# portable-op registration based on the PTE or an explicit select-ops list.
# Avoid feeding those cache entries into the top-level ExecuTorch configure,
# otherwise executorch_core auto-right-sizes MAX_KERNEL_NUM from the runner's
# placeholder/selective build inputs even though the runner also links
# quantized/cortex-m registration libraries.
set(_arm_runner_selective_cache_vars
    EXECUTORCH_SELECT_OPS_LIST EXECUTORCH_SELECT_OPS_MODEL
    EXECUTORCH_SELECT_OPS_YAML
)
foreach(_arm_runner_cache_var IN LISTS _arm_runner_selective_cache_vars)
  if(DEFINED CACHE{${_arm_runner_cache_var}})
    get_property(
      _arm_runner_cache_type
      CACHE ${_arm_runner_cache_var}
      PROPERTY TYPE
    )
    if(NOT _arm_runner_cache_type OR _arm_runner_cache_type STREQUAL
                                     "UNINITIALIZED"
    )
      set(_arm_runner_cache_type STRING)
    endif()
    set(_arm_runner_saved_type_${_arm_runner_cache_var}
        "${_arm_runner_cache_type}"
    )
    set(_arm_runner_saved_value_${_arm_runner_cache_var}
        "${${_arm_runner_cache_var}}"
    )
    set(_arm_runner_saved_defined_${_arm_runner_cache_var} TRUE)
    set(${_arm_runner_cache_var}
        ""
        CACHE ${_arm_runner_cache_type} "" FORCE
    )
  endif()
endforeach()

if(NOT DEFINED CACHE{MAX_KERNEL_NUM} AND NOT DEFINED MAX_KERNEL_NUM)
  set(MAX_KERNEL_NUM
      2000
      CACHE STRING
            "Maximum number of kernels registered by the standalone Arm runner"
  )
endif()

# Pull ExecuTorch in-tree so all required targets (delegates, kernels, runner
# util, etc.) are built from this checkout.
add_subdirectory(
  ${EXECUTORCH_ROOT} ${CMAKE_BINARY_DIR}/executorch EXCLUDE_FROM_ALL
)

if(ARM_EXECUTOR_RUNNER_SKIP_INSTALL_RULES)
  if(DEFINED _arm_runner_skip_install_rules)
    set(CMAKE_SKIP_INSTALL_RULES "${_arm_runner_skip_install_rules}")
  else()
    unset(CMAKE_SKIP_INSTALL_RULES)
  endif()
endif()

foreach(_arm_runner_cache_var IN LISTS _arm_runner_selective_cache_vars)
  if(_arm_runner_saved_defined_${_arm_runner_cache_var})
    set(${_arm_runner_cache_var}
        "${_arm_runner_saved_value_${_arm_runner_cache_var}}"
        CACHE ${_arm_runner_saved_type_${_arm_runner_cache_var}} "" FORCE
    )
  endif()
endforeach()

add_subdirectory(
  ${EXECUTORCH_ROOT}/examples/arm/executor_runner
  ${CMAKE_BINARY_DIR}/examples/arm/executor_runner
)
