# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

# Internal helper routines shared by Arm runner examples and superbuilds. These
# functions are not a stable public CMake API.

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
include(${EXECUTORCH_ROOT}/backends/arm/scripts/corstone_utils.cmake)
include(${EXECUTORCH_ROOT}/backends/arm/cmake/ArmEthosUSDK.cmake)

macro(arm_runner_require_python)
  if(NOT PYTHON_EXECUTABLE)
    find_package(
      Python3
      COMPONENTS Interpreter
      REQUIRED
    )
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
endmacro()

macro(arm_runner_add_standalone_executorch)
  if(DEFINED CMAKE_SKIP_INSTALL_RULES)
    set(_arm_runner_skip_install_rules_was_defined TRUE)
    set(_arm_runner_skip_install_rules "${CMAKE_SKIP_INSTALL_RULES}")
  else()
    set(_arm_runner_skip_install_rules_was_defined FALSE)
  endif()
  if(DEFINED CACHE{CMAKE_SKIP_INSTALL_RULES})
    set(_arm_runner_skip_install_rules_cache_was_defined TRUE)
    get_property(
      _arm_runner_skip_install_rules_cache_type
      CACHE CMAKE_SKIP_INSTALL_RULES
      PROPERTY TYPE
    )
    set(_arm_runner_skip_install_rules_cache "$CACHE{CMAKE_SKIP_INSTALL_RULES}")
  else()
    set(_arm_runner_skip_install_rules_cache_was_defined FALSE)
  endif()

  arm_runner_require_python()

  include(${EXECUTORCH_ROOT}/tools/cmake/common/preset.cmake)
  if(NOT DEFINED EXECUTORCH_BUILD_PRESET_FILE)
    set(EXECUTORCH_BUILD_PRESET_FILE
        "${EXECUTORCH_ROOT}/tools/cmake/preset/arm_baremetal.cmake"
        CACHE PATH "ExecuTorch build preset"
    )
  endif()
  load_build_preset()
  foreach(
    _arm_runner_option
    EXECUTORCH_BUILD_ARM_BAREMETAL EXECUTORCH_BUILD_CORTEX_M
    EXECUTORCH_BUILD_KERNELS_QUANTIZED EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL
  )
    set(${_arm_runner_option}
        ON
        CACHE BOOL "" FORCE
    )
  endforeach()
  set(EXECUTORCH_SKIP_ARM_EXECUTOR_RUNNER
      ON
      CACHE BOOL "" FORCE
  )
  set(MAX_KERNEL_NUM
      2000
      CACHE STRING "Maximum registered kernels"
  )

  set(_arm_runner_select_ops "${EXECUTORCH_SELECT_OPS_LIST}")
  set(EXECUTORCH_SELECT_OPS_LIST "")
  set(CMAKE_SKIP_INSTALL_RULES ON)
  add_subdirectory(
    ${EXECUTORCH_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/executorch EXCLUDE_FROM_ALL
  )
  if(_arm_runner_skip_install_rules_cache_was_defined)
    set(CMAKE_SKIP_INSTALL_RULES
        "${_arm_runner_skip_install_rules_cache}"
        CACHE "${_arm_runner_skip_install_rules_cache_type}" "" FORCE
    )
  else()
    unset(CMAKE_SKIP_INSTALL_RULES CACHE)
  endif()
  if(_arm_runner_skip_install_rules_was_defined)
    set(CMAKE_SKIP_INSTALL_RULES "${_arm_runner_skip_install_rules}")
  else()
    unset(CMAKE_SKIP_INSTALL_RULES)
  endif()
  unset(_arm_runner_skip_install_rules)
  unset(_arm_runner_skip_install_rules_was_defined)
  unset(_arm_runner_skip_install_rules_cache)
  unset(_arm_runner_skip_install_rules_cache_type)
  unset(_arm_runner_skip_install_rules_cache_was_defined)
  set(EXECUTORCH_SELECT_OPS_LIST "${_arm_runner_select_ops}")
  include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
endmacro()

function(arm_runner_add_minimal_executable)
  set(one_value_args TARGET SOURCE OPS_PREFIX)
  set(multi_value_args COMPILE_DEFINITIONS)
  cmake_parse_arguments(
    ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  if(NOT ARG_TARGET
     OR NOT ARG_SOURCE
     OR NOT ARG_OPS_PREFIX
  )
    message(FATAL_ERROR "TARGET, SOURCE, and OPS_PREFIX are required.")
  endif()

  add_executable(
    ${ARG_TARGET}
    ${ARG_SOURCE}
    ${EXECUTORCH_ROOT}/examples/arm/common/arm_memory_allocator.cpp
  )
  target_include_directories(
    ${ARG_TARGET}
    PRIVATE ${EXECUTORCH_ROOT}
            ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
            ${EXECUTORCH_ROOT}/examples/arm/common ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_compile_definitions(
    ${ARG_TARGET}
    PRIVATE
      C10_USING_CUSTOM_GENERATED_MACROS
      ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
      ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE}
      ${ARG_COMPILE_DEFINITIONS}
  )
  arm_runner_configure_model(
    TARGET
    ${ARG_TARGET}
    PTE_FILE
    "${ET_PTE_FILE_PATH}"
    MODEL_PTE_ADDR
    "${ET_MODEL_PTE_ADDR}"
    MODEL_PTE_SIZE
    "${ET_MODEL_PTE_SIZE}"
  )
  arm_runner_configure_linker_script(
    TARGET ${ARG_TARGET} SYSTEM_CONFIG "${SYSTEM_CONFIG}"
  )
  arm_runner_link_minimal_specs(${ARG_TARGET})
  arm_runner_link_registration_libraries(
    TARGET
    ${ARG_TARGET}
    BASE_LIBS
    extension_runner_util
    ethosu_target_init
    REGISTRATION_LIBS
    executorch
    executorch_delegate_ethos_u
    ${ARG_OPS_PREFIX}_portable_ops
    ${ARG_OPS_PREFIX}_quantized_ops
    ${ARG_OPS_PREFIX}_cortex_m_ops
    NORMAL_LIBS
    cortex_m_kernels
    quantized_kernels
    portable_kernels
    kernels_util_all_deps
  )
  target_link_options(${ARG_TARGET} PRIVATE LINKER:-Map=${ARG_TARGET}.map)
endfunction()

function(arm_runner_define_cache_options)
  set(one_value_args METHOD_ALLOCATOR_SIZE)
  cmake_parse_arguments(ARG "" "${one_value_args}" "" ${ARGN})
  set(ET_PTE_FILE_PATH
      ""
      CACHE PATH "Path to an ExecuTorch model PTE"
  )
  set(ET_MODEL_PTE_ADDR
      ""
      CACHE STRING "Address of a memory-mapped PTE"
  )
  set(ET_MODEL_PTE_SIZE
      ""
      CACHE STRING "Size in bytes of a memory-mapped PTE"
  )
  set(EXECUTORCH_SELECT_OPS_LIST
      ""
      CACHE STRING "Explicit operator selection"
  )
  set(ETHOS_SDK_PATH
      "${EXECUTORCH_ROOT}/examples/arm/arm-scratch/ethos-u"
      CACHE PATH "Path to the Ethos-U bare-metal SDK"
  )
  set(SYSTEM_CONFIG
      "Ethos_U55_High_End_Embedded"
      CACHE STRING "Ethos-U system configuration"
  )
  set(MEMORY_MODE
      "Shared_Sram"
      CACHE STRING "Ethos-U memory mode"
  )
  if(MEMORY_MODE MATCHES "^Dedicated_Sram($|_)")
    set(_scratch_size 0x4000000)
  else()
    set(_scratch_size 0x200000)
  endif()
  if(NOT DEFINED ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
    set(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE
        ${_scratch_size}
        PARENT_SCOPE
    )
  endif()
  set(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE
      "${ARG_METHOD_ALLOCATOR_SIZE}"
      CACHE STRING "Persistent method allocator size"
  )
endfunction()

function(arm_runner_validate_model_source)
  cmake_parse_arguments(ARG "ALLOW_SEMIHOSTING" "" "" ${ARGN})
  if(NOT (ARG_ALLOW_SEMIHOSTING AND SEMIHOSTING)
     AND NOT ET_MODEL_PTE_ADDR
     AND "${ET_PTE_FILE_PATH}" STREQUAL ""
  )
    message(FATAL_ERROR "Set ET_PTE_FILE_PATH or ET_MODEL_PTE_ADDR.")
  endif()
  if(NOT "${ET_PTE_FILE_PATH}" STREQUAL "" AND NOT EXISTS "${ET_PTE_FILE_PATH}")
    message(FATAL_ERROR "ET_PTE_FILE_PATH does not exist: ${ET_PTE_FILE_PATH}")
  endif()
  if(ET_MODEL_PTE_ADDR AND NOT ET_MODEL_PTE_SIZE)
    message(
      WARNING
        "ET_MODEL_PTE_SIZE is unset; using the legacy 0x10000000 upper bound."
    )
    set(ET_MODEL_PTE_SIZE
        0x10000000
        PARENT_SCOPE
    )
  endif()
  if(ET_MODEL_PTE_SIZE AND NOT ET_MODEL_PTE_ADDR)
    message(FATAL_ERROR "ET_MODEL_PTE_SIZE requires ET_MODEL_PTE_ADDR.")
  endif()
endfunction()

#[[
Configure how an ExecuTorch model is made available to a runner target.

Arguments:
  TARGET: Existing runner target to configure.
  PTE_FILE: PTE file to convert into a generated model_pte.h header.
  MODEL_PTE_ADDR: Address of a PTE already present in target memory. This takes
    precedence over PTE_FILE.
  MODEL_PTE_SIZE: Size in bytes of the PTE at MODEL_PTE_ADDR. If omitted, the
    legacy 0x10000000 upper bound is used.
  PUBLIC: Propagate the selected model compile definition to dependants.
]]
function(arm_runner_configure_model)
  set(options PUBLIC)
  set(one_value_args TARGET PTE_FILE MODEL_PTE_ADDR MODEL_PTE_SIZE)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "" ${ARGN})
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "TARGET is required.")
  endif()

  if(ARG_PUBLIC)
    set(_scope PUBLIC)
  else()
    set(_scope PRIVATE)
  endif()

  if(ARG_MODEL_PTE_ADDR)
    target_compile_definitions(
      ${ARG_TARGET} ${_scope} ET_MODEL_PTE_ADDR=${ARG_MODEL_PTE_ADDR}
                    ET_MODEL_PTE_SIZE=${ARG_MODEL_PTE_SIZE}
    )
  elseif(NOT "${ARG_PTE_FILE}" STREQUAL "")
    if(Python3_EXECUTABLE)
      set(_python_executable "${Python3_EXECUTABLE}")
    elseif(PYTHON_EXECUTABLE)
      set(_python_executable "${PYTHON_EXECUTABLE}")
    else()
      find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED
      )
      set(_python_executable "${Python3_EXECUTABLE}")
    endif()
    set(_model_header "${CMAKE_CURRENT_BINARY_DIR}/model_pte.h")
    add_custom_command(
      OUTPUT ${_model_header}
      COMMAND
        ${_python_executable}
        ${EXECUTORCH_ROOT}/examples/arm/common/pte_to_header.py --pte
        ${ARG_PTE_FILE} --outdir ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${ARG_PTE_FILE}
    )
    add_custom_target(${ARG_TARGET}_model_header DEPENDS ${_model_header})
    add_dependencies(${ARG_TARGET} ${ARG_TARGET}_model_header)
    target_compile_definitions(${ARG_TARGET} ${_scope} ET_COMPILED_PTE)
  endif()
endfunction()

function(arm_runner_configure_ethos_u_platform)
  set(one_value_args SDK_PATH SYSTEM_CONFIG MEMORY_MODE)
  cmake_parse_arguments(ARG "" "${one_value_args}" "" ${ARGN})
  if(NOT ARG_SDK_PATH
     OR NOT ARG_SYSTEM_CONFIG
     OR NOT ARG_MEMORY_MODE
  )
    message(
      FATAL_ERROR "SDK_PATH, SYSTEM_CONFIG, and MEMORY_MODE are required."
    )
  endif()

  arm_ethos_u_default_fetch("${ARG_SDK_PATH}" _fetch_ethos_u_default)
  option(FETCH_ETHOS_U_CONTENT "Fetch Ethos-U dependencies"
         ${_fetch_ethos_u_default}
  )
  arm_ensure_ethos_u_content(
    "${ARG_SDK_PATH}" "${EXECUTORCH_ROOT}" ${FETCH_ETHOS_U_CONTENT}
  )
  add_corstone_subdirectory(${ARG_SYSTEM_CONFIG} ${ARG_SDK_PATH})
  configure_timing_adapters(${ARG_SYSTEM_CONFIG} ${ARG_MEMORY_MODE})
  foreach(_platform_variable TARGET_BOARD ETHOSU_MODEL ETHOSU_ARENA)
    if(DEFINED ${_platform_variable})
      set(${_platform_variable}
          "${${_platform_variable}}"
          PARENT_SCOPE
      )
    endif()
  endforeach()

  if(NOT CMAKE_SKIP_INSTALL_RULES AND TARGET ethosu_core_driver)
    get_property(
      _ethosu_core_driver_exported GLOBAL
      PROPERTY ET_ETHOSU_CORE_DRIVER_EXPORTED
    )
    if(NOT _ethosu_core_driver_exported)
      install(
        TARGETS ethosu_core_driver
        EXPORT ExecuTorchTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )
      set_property(GLOBAL PROPERTY ET_ETHOSU_CORE_DRIVER_EXPORTED TRUE)
    endif()
  endif()
endfunction()

# Verify that targets required for building the executor runner are present.
function(arm_runner_require_baremetal_targets)
  if(NOT TARGET extension_runner_util)
    message(
      FATAL_ERROR
        "extension_runner_util target missing. Configure ExecuTorch (or the standalone runner) with EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON."
    )
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

  PRIM_OPS: Build a selective prim ops library instead of a codegen operators lib.

The lib will always be created, even when no operators are selected.
OP_LIST and OPS_FROM_MODEL can be used additively.
If neither OP_LIST nor OPS_FROM_MODEL is set, the generated registration
library is empty. INCLUDE_ALL_OPS explicitly builds a full registration library.
]]
function(arm_runner_create_selected_ops_lib)
  # Parse arguments
  set(options INCLUDE_ALL_OPS PRIM_OPS)
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

  if(NOT ARG_PRIM_OPS
     AND NOT ARG_FUNCTIONS_YAML
     AND NOT ARG_CUSTOM_OPS_YAML
  )
    message(
      FATAL_ERROR
        "${_arm_runner_create_selected_ops_lib_context} requires FUNCTIONS_YAML or CUSTOM_OPS_YAML unless PRIM_OPS is set."
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
  set(_arm_runner_include_all_ops OFF)
  if(ARG_INCLUDE_ALL_OPS)
    set(_arm_runner_include_all_ops ON)
    list(APPEND _arm_runner_selected_ops_args INCLUDE_ALL_OPS "ON")
  endif()
  gen_selected_ops(${_arm_runner_selected_ops_args})

  if(ARG_PRIM_OPS)
    set(_arm_runner_prim_ops_args
        LIB_NAME "${ARG_LIB_NAME}" SELECTED_OPS_YAML
        "${gen_selected_ops_output_yaml}" DEPS ${ARG_DEPS}
    )
    if(_arm_runner_include_all_ops)
      list(APPEND _arm_runner_prim_ops_args INCLUDE_ALL_OPS)
    endif()
    gen_selected_prim_ops_lib(${_arm_runner_prim_ops_args})
    return()
  endif()

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

function(arm_runner_create_default_selected_ops_libs)
  set(one_value_args PREFIX SUFFIX OP_LIST OPS_FROM_MODEL DTYPE_SELECTIVE_BUILD
                     OUT_LIBS
  )
  set(multi_value_args DEPS)
  cmake_parse_arguments(
    ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  if(NOT ARG_PREFIX)
    message(FATAL_ERROR "PREFIX is required.")
  endif()

  if(ARG_DEPS)
    set(_portable_deps ${ARG_DEPS})
    set(_quantized_deps ${ARG_DEPS})
    set(_cortex_m_deps ${ARG_DEPS})
  else()
    set(_portable_deps executorch)
    set(_quantized_deps executorch_core)
    set(_cortex_m_deps executorch)
  endif()

  set(_portable_ops "${ARG_PREFIX}_portable_ops${ARG_SUFFIX}")
  set(_quantized_ops "${ARG_PREFIX}_quantized_ops${ARG_SUFFIX}")
  set(_cortex_m_ops "${ARG_PREFIX}_cortex_m_ops${ARG_SUFFIX}")

  arm_runner_create_selected_ops_lib(
    LIB_NAME
    ${_portable_ops}
    FUNCTIONS_YAML
    "${EXECUTORCH_ROOT}/kernels/portable/functions.yaml"
    OP_LIST
    "${ARG_OP_LIST}"
    OPS_FROM_MODEL
    "${ARG_OPS_FROM_MODEL}"
    DTYPE_SELECTIVE_BUILD
    "${ARG_DTYPE_SELECTIVE_BUILD}"
    KERNEL_LIBS
    portable_kernels
    DEPS
    ${_portable_deps}
  )
  arm_runner_create_selected_ops_lib(
    LIB_NAME
    ${_quantized_ops}
    CUSTOM_OPS_YAML
    "${EXECUTORCH_ROOT}/kernels/quantized/quantized.yaml"
    OP_LIST
    "${ARG_OP_LIST}"
    OPS_FROM_MODEL
    "${ARG_OPS_FROM_MODEL}"
    KERNEL_LIBS
    quantized_kernels
    DEPS
    ${_quantized_deps}
  )
  arm_runner_create_selected_ops_lib(
    LIB_NAME
    ${_cortex_m_ops}
    CUSTOM_OPS_YAML
    "${EXECUTORCH_ROOT}/backends/cortex_m/ops/operators.yaml"
    OP_LIST
    "${ARG_OP_LIST}"
    OPS_FROM_MODEL
    "${ARG_OPS_FROM_MODEL}"
    KERNEL_LIBS
    cortex_m_kernels
    DEPS
    ${_cortex_m_deps}
  )
  if(ARG_OUT_LIBS)
    set(${ARG_OUT_LIBS}
        ${_portable_ops} ${_quantized_ops} ${_cortex_m_ops}
        PARENT_SCOPE
    )
  endif()
endfunction()

# Ensure a runner target emits its binary to a predictable location. Uses
# FALLBACK_DIR when TARGET_NAME has no runtime output directory set, and also
# fills per-configuration runtime output directories for multi-config generators
# when they are unset.
function(arm_runner_link_registration_libraries)
  set(one_value_args TARGET SCOPE)
  set(multi_value_args BASE_LIBS REGISTRATION_LIBS NORMAL_LIBS SUPPRESS_LIBS)
  cmake_parse_arguments(
    ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "TARGET is required.")
  endif()
  if(NOT ARG_SCOPE)
    set(ARG_SCOPE PRIVATE)
  endif()

  foreach(_registration_target IN LISTS ARG_REGISTRATION_LIBS ARG_SUPPRESS_LIBS)
    if(TARGET ${_registration_target})
      set_property(
        TARGET ${_registration_target} PROPERTY INTERFACE_LINK_OPTIONS ""
      )
    endif()
  endforeach()

  target_link_libraries(
    ${ARG_TARGET}
    ${ARG_SCOPE}
    -Wl,--start-group
    ${ARG_BASE_LIBS}
    -Wl,--whole-archive
    ${ARG_REGISTRATION_LIBS}
    -Wl,--no-whole-archive
    ${ARG_NORMAL_LIBS}
    -Wl,--end-group
  )
endfunction()

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

#[[
Preprocess and attach the Corstone linker script for a runner target.

Arguments:
  TARGET: Existing runner target to configure.
  SYSTEM_CONFIG: Ethos-U system configuration used to select the Corstone FVP.
  OUTPUT_NAME: Optional basename for the generated linker script.
]]
function(arm_runner_configure_linker_script)
  set(one_value_args TARGET SYSTEM_CONFIG OUTPUT_NAME)
  cmake_parse_arguments(ARG "" "${one_value_args}" "" ${ARGN})

  if(NOT ARG_TARGET OR NOT ARG_SYSTEM_CONFIG)
    message(
      FATAL_ERROR
        "arm_runner_configure_linker_script requires TARGET and SYSTEM_CONFIG."
    )
  endif()
  verify_targets_exist(
    CONTEXT arm_runner_configure_linker_script TARGETS ${ARG_TARGET}
  )

  get_corstone_linker_script(_linker_script "${ARG_SYSTEM_CONFIG}")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LINK_FILE_EXT ld)
    set(LINK_FILE_OPTION "-T")
    set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
  else()
    message(
      FATAL_ERROR
        "arm_runner_configure_linker_script only supports the GNU compiler."
    )
  endif()

  if(NOT ARG_OUTPUT_NAME)
    set(ARG_OUTPUT_NAME "${ARG_TARGET}_linker_script")
  endif()
  set(_linker_script_out
      "${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT_NAME}.${LINK_FILE_EXT}"
  )
  set(_preprocessor_options ${COMPILER_PREPROCESSOR_OPTIONS})
  foreach(_define IN ITEMS HEAP_SIZE STACK_SIZE ETHOSU_MODEL ETHOSU_ARENA)
    if(DEFINED ${_define})
      list(APPEND _preprocessor_options "-D${_define}=${${_define}}")
    endif()
  endforeach()

  execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${_preprocessor_options} -o
            ${_linker_script_out} ${_linker_script} COMMAND_ERROR_IS_FATAL ANY
  )
  target_link_options(
    ${ARG_TARGET} PRIVATE "${LINK_FILE_OPTION}" "${_linker_script_out}"
  )
endfunction()
