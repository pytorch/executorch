# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

# Helper routines shared by the standalone runner and any superbuild that reuses
# the runner targets.

get_filename_component(
  _arm_runner_executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE
)
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT "${_arm_runner_executorch_root}")
endif()

# Make sure codegen function used below is loaded.
if(NOT COMMAND gen_selected_ops)
  include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
endif()

# Verify that targets required for building the executor runner are present.
function(arm_runner_require_baremetal_targets)
  if(NOT TARGET extension_runner_util)
    message(
      FATAL_ERROR
        "extension_runner_util target missing. Configure ExecuTorch (or the standalone runner) with EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON."
    )
  endif()

endfunction()

#[[
Create a selected operator registration library for one operator family.

Arguments:
  LIB_NAME: Name of the generated registration library target.
	example: arm_cortex_m_ops_lib

  FUNCTIONS_YAML: Portable ATen operator YAML used to generate bindings.
	example: ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml

  CUSTOM_OPS_YAML: Custom/backend operator YAML used to generate bindings.
	example: ${EXECUTORCH_ROOT}/backends/cortex_m/ops/operators.yaml

  OP_LIST: Operators to select.
	example: ${EXECUTORCH_SELECT_OPS_LIST}

  OPS_FROM_MODEL: ExecuTorch model file used to select operators.
	example: ${ET_PTE_FILE_PATH}

  DTYPE_SELECTIVE_BUILD: Enables dtype filtering for operator libraries that support it.
	example: ${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}

  KERNEL_LIBS: Kernel implementation libraries used by the generated target.
	example: cortex_m_kernels

  DEPS: Additional libraries required by the generated registration target.
	example: executorch

  INCLUDE_ALL_OPS: Generate a non-selective registration library for the
	selected operator family.

The lib will always be created, even when no operators are selected.
OP_LIST and OPS_FROM_MODEL can be used additively.
]]
function(arm_runner_create_selected_ops_lib)
  # Parse arguments
  set(options INCLUDE_ALL_OPS)
  set(one_value_args LIB_NAME FUNCTIONS_YAML CUSTOM_OPS_YAML OP_LIST
                     OPS_FROM_MODEL DTYPE_SELECTIVE_BUILD
  )
  set(multi_value_args KERNEL_LIBS DEPS)
  cmake_parse_arguments(
    ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  # Validate and normalize arguments
  if(ARG_LIB_NAME)
    set(_arm_runner_create_selected_ops_lib_context
        "arm_runner_create_selected_ops_lib(${ARG_LIB_NAME})"
    )
  else()
    set(_arm_runner_create_selected_ops_lib_context
        "arm_runner_create_selected_ops_lib"
    )
  endif()

  if(ARG_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
        "${_arm_runner_create_selected_ops_lib_context} got unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}."
    )
  endif()

  if(NOT ARG_LIB_NAME)
    message(
      FATAL_ERROR
        "${_arm_runner_create_selected_ops_lib_context} requires LIB_NAME."
    )
  endif()

  if(NOT ARG_FUNCTIONS_YAML AND NOT ARG_CUSTOM_OPS_YAML)
    message(
      FATAL_ERROR
        "${_arm_runner_create_selected_ops_lib_context} requires FUNCTIONS_YAML or CUSTOM_OPS_YAML."
    )
  endif()

  if(ARG_DTYPE_SELECTIVE_BUILD)
    executorch_load_build_variables()
  endif()

  verify_targets_exist(
    CONTEXT "${_arm_runner_create_selected_ops_lib_context}" TARGETS
    ${ARG_KERNEL_LIBS} ${ARG_DEPS}
  )

  # Generate selected operator list file.
  set(_arm_runner_selected_ops_args
      LIB_NAME
      "${ARG_LIB_NAME}"
      ROOT_OPS
      "${ARG_OP_LIST}"
      OPS_FROM_MODEL
      "${ARG_OPS_FROM_MODEL}"
      DTYPE_SELECTIVE_BUILD
      "${ARG_DTYPE_SELECTIVE_BUILD}"
  )
  if(ARG_INCLUDE_ALL_OPS)
    list(APPEND _arm_runner_selected_ops_args INCLUDE_ALL_OPS "ON")
  endif()
  gen_selected_ops(${_arm_runner_selected_ops_args})

  # Codegen for operator library.
  set(_arm_ops_binding_args LIB_NAME "${ARG_LIB_NAME}")
  if(ARG_FUNCTIONS_YAML)
    list(APPEND _arm_ops_binding_args FUNCTIONS_YAML "${ARG_FUNCTIONS_YAML}")
  endif()
  if(ARG_CUSTOM_OPS_YAML)
    list(APPEND _arm_ops_binding_args CUSTOM_OPS_YAML "${ARG_CUSTOM_OPS_YAML}")
  endif()
  if(ARG_DTYPE_SELECTIVE_BUILD)
    list(APPEND _arm_ops_binding_args DTYPE_SELECTIVE_BUILD
         "${ARG_DTYPE_SELECTIVE_BUILD}"
    )
  endif()
  generate_bindings_for_kernels(${_arm_ops_binding_args})

  # Finally, build operator library.
  gen_operators_lib(
    LIB_NAME
    "${ARG_LIB_NAME}"
    KERNEL_LIBS
    ${ARG_KERNEL_LIBS}
    DEPS
    ${ARG_DEPS}
    DTYPE_SELECTIVE_BUILD
    "${ARG_DTYPE_SELECTIVE_BUILD}"
  )
endfunction()

# Ensure a runner target emits its binary to a predictable location. Uses
# FALLBACK_DIR when TARGET_NAME has no runtime output directory set, and also
# fills per-configuration runtime output directories for multi-config generators
# when they are unset.
function(arm_runner_configure_runtime_output TARGET_NAME FALLBACK_DIR)
  if(NOT TARGET ${TARGET_NAME})
    return()
  endif()

  get_target_property(_base_runtime_dir ${TARGET_NAME} RUNTIME_OUTPUT_DIRECTORY)
  if(NOT _base_runtime_dir
     OR _base_runtime_dir STREQUAL "_base_runtime_dir-NOTFOUND"
     OR "${_base_runtime_dir}" STREQUAL ""
  )
    set_target_properties(
      ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${FALLBACK_DIR}"
    )
    set(_base_runtime_dir "${FALLBACK_DIR}")
  endif()

  if(CMAKE_CONFIGURATION_TYPES)
    foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${_cfg} _cfg_upper)
      set(_cfg_prop "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}")
      get_target_property(_cfg_dir ${TARGET_NAME} ${_cfg_prop})
      if(NOT _cfg_dir
         OR _cfg_dir STREQUAL "_cfg_dir-NOTFOUND"
         OR "${_cfg_dir}" STREQUAL ""
      )
        set_target_properties(
          ${TARGET_NAME} PROPERTIES ${_cfg_prop} "${_base_runtime_dir}/${_cfg}"
        )
      endif()
    endforeach()
  endif()
endfunction()

function(verify_targets_exist)
  cmake_parse_arguments(ARG "" "CONTEXT" "TARGETS" ${ARGN})

  set(_targets ${ARG_TARGETS} ${ARG_UNPARSED_ARGUMENTS})
  if(NOT ARG_CONTEXT)
    set(ARG_CONTEXT "verify_targets_exist")
  endif()

  foreach(_target IN LISTS _targets)
    if(NOT TARGET ${_target})
      message(FATAL_ERROR "${ARG_CONTEXT} requires missing target ${_target}.")
    endif()
  endforeach()
endfunction()

# Link the provided target with minimal specs. This minimizes code size, but
# comes with limited support. Notably, printing with %zu is not supported.
function(arm_runner_link_minimal_specs TARGET_NAME)
  verify_targets_exist(
    CONTEXT arm_runner_link_minimal_specs TARGETS ${TARGET_NAME}
  )
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(
      ${TARGET_NAME} PRIVATE --specs=nano.specs -u _printf_float
    )
  endif()
endfunction()
