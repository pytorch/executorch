# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(DIRECTORY)

# Stable CMake API for in-tree and external Arm runner consumers. Incompatible
# changes require a migration path for existing consumers.
# cmake-format: off
set(ARM_RUNNER_UTILS_API_COMMANDS
    arm_runner_add_minimal_executable
    arm_runner_add_standalone_executorch
    arm_runner_configure_ethos_u_platform
    arm_runner_configure_linker_script
    arm_runner_configure_model
    arm_runner_configure_runtime_output
    arm_runner_create_default_selected_ops_libs
    arm_runner_create_selected_ops_lib
    arm_runner_define_cache_options
    arm_runner_link_minimal_specs
    arm_runner_link_registration_libraries
    arm_runner_require_baremetal_targets
    arm_runner_require_python
    arm_runner_validate_model_source
)

# KIND distinguishes functions from macros; SIGNATURE records the stable call
# interface.
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_add_minimal_executable function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_add_minimal_executable
    "arm_runner_add_minimal_executable(*, TARGET, SOURCE, OPS_PREFIX, COMPILE_DEFINITIONS=())"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_add_standalone_executorch macro)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_add_standalone_executorch
    "arm_runner_add_standalone_executorch()"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_configure_ethos_u_platform function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_configure_ethos_u_platform
    "arm_runner_configure_ethos_u_platform(*, SDK_PATH, SYSTEM_CONFIG, MEMORY_MODE)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_configure_linker_script function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_configure_linker_script
    "arm_runner_configure_linker_script(*, TARGET, SYSTEM_CONFIG, OUTPUT_NAME=None)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_configure_model function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_configure_model
    "arm_runner_configure_model(*, TARGET, PTE_FILE=None, MODEL_PTE_ADDR=None, MODEL_PTE_SIZE=None, PUBLIC=False)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_configure_runtime_output function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_configure_runtime_output
    "arm_runner_configure_runtime_output(TARGET_NAME, FALLBACK_DIR)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_create_default_selected_ops_libs
    function
)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_create_default_selected_ops_libs
    "arm_runner_create_default_selected_ops_libs(*, PREFIX, SUFFIX=None, OP_LIST=None, OPS_FROM_MODEL=None, DTYPE_SELECTIVE_BUILD=None, OUT_LIBS=None, DEPS=())"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_create_selected_ops_lib function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_create_selected_ops_lib
    "arm_runner_create_selected_ops_lib(*, LIB_NAME, FUNCTIONS_YAML=None, CUSTOM_OPS_YAML=None, OP_LIST=None, OPS_FROM_MODEL=None, DTYPE_SELECTIVE_BUILD=None, KERNEL_LIBS=(), DEPS=(), INCLUDE_ALL_OPS=False, PRIM_OPS=False)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_define_cache_options function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_define_cache_options
    "arm_runner_define_cache_options(*, METHOD_ALLOCATOR_SIZE=None)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_link_minimal_specs function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_link_minimal_specs
    "arm_runner_link_minimal_specs(TARGET_NAME)"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_link_registration_libraries function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_link_registration_libraries
    "arm_runner_link_registration_libraries(*, TARGET, SCOPE=None, BASE_LIBS=(), REGISTRATION_LIBS=(), NORMAL_LIBS=(), SUPPRESS_LIBS=())"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_require_baremetal_targets function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_require_baremetal_targets
    "arm_runner_require_baremetal_targets()"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_require_python macro)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_require_python
    "arm_runner_require_python()"
)
set(ARM_RUNNER_UTILS_API_KIND_arm_runner_validate_model_source function)
set(ARM_RUNNER_UTILS_API_SIGNATURE_arm_runner_validate_model_source
    "arm_runner_validate_model_source(*, ALLOW_SEMIHOSTING=False)"
)
# cmake-format: on

# Validate the API metadata before loading the implementation.
foreach(_arm_runner_api_command IN LISTS ARM_RUNNER_UTILS_API_COMMANDS)
  if(NOT DEFINED ARM_RUNNER_UTILS_API_KIND_${_arm_runner_api_command}
     OR NOT DEFINED ARM_RUNNER_UTILS_API_SIGNATURE_${_arm_runner_api_command}
  )
    message(
      FATAL_ERROR
        "Arm runner CMake API metadata is missing for ${_arm_runner_api_command}."
    )
  endif()
  if(NOT ARM_RUNNER_UTILS_API_KIND_${_arm_runner_api_command} MATCHES
     "^(function|macro)$"
  )
    message(
      FATAL_ERROR
        "Arm runner CMake API kind must be function or macro for ${_arm_runner_api_command}."
    )
  endif()
  if(NOT ARM_RUNNER_UTILS_API_SIGNATURE_${_arm_runner_api_command} MATCHES
     "^${_arm_runner_api_command}\\("
  )
    message(
      FATAL_ERROR
        "Arm runner CMake API signature does not match ${_arm_runner_api_command}."
    )
  endif()
endforeach()

if(ARM_RUNNER_UTILS_API_METADATA_ONLY)
  if(ARM_RUNNER_UTILS_API_EMIT_MANIFEST)
    foreach(_arm_runner_api_command IN LISTS ARM_RUNNER_UTILS_API_COMMANDS)
      set(_kind "${ARM_RUNNER_UTILS_API_KIND_${_arm_runner_api_command}}")
      set(_signature
          "${ARM_RUNNER_UTILS_API_SIGNATURE_${_arm_runner_api_command}}"
      )
      message(
        "ARM_RUNNER_UTILS_API|${_arm_runner_api_command}|${_kind}|${_signature}"
      )
    endforeach()
  endif()
  return()
endif()

get_cmake_property(_arm_runner_commands_before_include COMMANDS)
include(${CMAKE_CURRENT_LIST_DIR}/ArmRunnerUtilsInternal.cmake)

foreach(_arm_runner_api_command IN LISTS ARM_RUNNER_UTILS_API_COMMANDS)
  if(NOT COMMAND ${_arm_runner_api_command})
    message(
      FATAL_ERROR
        "Arm runner CMake API requires missing command ${_arm_runner_api_command}."
    )
  endif()
endforeach()

get_cmake_property(_arm_runner_commands COMMANDS)
list(REMOVE_ITEM _arm_runner_commands ${_arm_runner_commands_before_include})
list(FILTER _arm_runner_commands INCLUDE REGEX "^arm_runner_")
foreach(_arm_runner_command IN LISTS _arm_runner_commands)
  if(NOT _arm_runner_command IN_LIST ARM_RUNNER_UTILS_API_COMMANDS)
    message(
      FATAL_ERROR
        "Arm runner CMake command ${_arm_runner_command} is not declared in ARM_RUNNER_UTILS_API_COMMANDS. Add it to the API or use the _arm_runner_ prefix for a private helper."
    )
  endif()
endforeach()

unset(_arm_runner_api_command)
unset(_arm_runner_command)
unset(_arm_runner_commands)
unset(_arm_runner_commands_before_include)
